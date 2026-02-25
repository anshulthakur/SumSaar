import os
import json
import re
from typing import List
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from openai import OpenAI

import logging

logger = logging.getLogger(__name__)

import django
from django.conf import settings
if not settings.configured:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "sumsaar.webserver.webserver.settings")
    django.setup()

from sumsaar.settings import OLLAMA_URL, SUMMARY_LLM
from chitrapat.models import RawArticle, StoryCluster, SynthesizedArticle, ArticleVector
from pgvector.django import CosineDistance

# --- LLM Definitions ---

class LLMArticle(BaseModel):
    title: str
    content: str
    keywords: List[str] = []

class LLM(object):
    def __init__(self, host="http://localhost:11434", **kwargs):
        base_url = OLLAMA_URL
        if '/v1' not in base_url:
            base_url = f"{base_url}/v1"
        self.client = OpenAI(
            base_url=base_url,
            api_key='ollama',
            timeout=5*60,
            )
        self.system_prompt = ''

    def clean_control_chars(self, s: str) -> str:
        return re.sub(r'[\x00-\x1F]', '', s)

    def generate(self, prompt, json_mode=True):
        try:
            extra_body = {
                'keep_alive': '10m',
                'options': {
                    'num_ctx': 8196*2,
                    'repeat_last_n': 64,
                    "top_k" : 0,
                    "min_p" : 0.1,
                    "mirostat": 0,
                    "repeat_penalty": 1.05,
                    "num_predict": 1024*8,
                },
            }
            
            kwargs = {
                'model': SUMMARY_LLM,
                'messages': [
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                'temperature': 0.3,
                'top_p': 0.8,
                'seed': 0,
                'extra_body': extra_body
            }

            if json_mode:
                kwargs['response_format'] = LLMArticle.model_json_schema()

            response = self.client.chat.completions.create(**kwargs)
            content = self.clean_control_chars(response.choices[0].message.content)
            
            if json_mode:
                return LLMArticle.model_validate_json(content)
            return content
        except Exception as e:
            logger.error(f'LLM Error: {e}')
            if json_mode:
                return LLMArticle(title='Error', content='Failed to generate content.')
            return "Error generating content."

class FactExtractor(LLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.system_prompt = ("Identify and extract the key facts, figures, dates, and distinct events from the text. "
                              "Present them as a concise list of bullet points. "
                              "Focus on hard information (who, what, when, where, why, numbers). "
                              "Ignore opinions and fluff.")

class StoryUpdater(LLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.system_prompt = ("You are a news editor maintaining a developing story. "
                              "I will provide the 'Current Story' and a list of 'New Facts'. "
                              "Update the story to incorporate the new facts. "
                              "If a new fact contradicts old info, overwrite it. "
                              "Do not delete relevant context. "
                              "IMPORTANT: Return JSON with title, content, keywords.")

class Summarizer(LLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.system_prompt = ("Write a comprehensive news report based on the provided input. "
                              "Ensure clarity, coherence, and factual accuracy. "
                              "IMPORTANT: Return JSON with title, content, keywords.")

# --- Core Pipeline Logic ---

def get_embedding(text):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(text[:2000]) # Truncate to avoid token limits

def process_incoming_article(raw_article_id):
    """
    Main pipeline entry point for a single article.
    1. Embed
    2. Search
    3. Synthesize (Create or Update)
    """
    try:
        raw_article = RawArticle.objects.get(id=raw_article_id)
    except RawArticle.DoesNotExist:
        logger.error(f"RawArticle {raw_article_id} not found.")
        return

    title = raw_article.source_data.get('title', '')
    content = raw_article.source_data.get('content', '')
    text_for_embedding = f"{title} {content}"
    
    logger.info(f"Processing: {title}")

    # 1. Generate Embedding
    embedding = get_embedding(text_for_embedding)

    # 2. Vector Search (Find nearest neighbor)
    # Threshold: 0.25 distance ~= 0.75 similarity
    match_threshold = 0.25 
    
    nearest_vector = ArticleVector.objects.annotate(
        distance=CosineDistance('embedding', embedding)
    ).order_by('distance').first()

    matched_article = None
    if nearest_vector and nearest_vector.distance < match_threshold:
        matched_article = nearest_vector.article
        logger.info(f"Match found: {matched_article.headline} (Dist: {nearest_vector.distance:.4f})")
    else:
        logger.info("No match found. Creating new story.")

    # 3. Synthesis
    if matched_article:
        # --- UPDATE FLOW ---
        fact_extractor = FactExtractor()
        story_updater = StoryUpdater()

        # Extract facts from new article
        facts = fact_extractor.generate(content, json_mode=False)
        
        # Update existing story
        prompt = f"Current Story:\n{matched_article.content}\n\nNew Facts:\n{facts}"
        updated_content = story_updater.generate(prompt, json_mode=True)
        
        # Save updates
        matched_article.headline = updated_content.title
        matched_article.content = updated_content.content
        matched_article.facts_timeline.append(facts) # Append atomic facts
        matched_article.sources.append(raw_article.url)
        matched_article.save()
        
        # Update embedding to reflect new content
        new_embedding = get_embedding(f"{updated_content.title} {updated_content.content}")
        nearest_vector.embedding = new_embedding
        nearest_vector.save()

    else:
        # --- NEW STORY FLOW ---
        summarizer = Summarizer()
        summary = summarizer.generate(content, json_mode=True)
        
        cluster = StoryCluster.objects.create(title=summary.title)
        
        new_article = SynthesizedArticle.objects.create(
            cluster=cluster,
            headline=summary.title,
            content=summary.content,
            sources=[raw_article.url],
            facts_timeline=[content] # Initial fact base is the content itself
        )
        
        ArticleVector.objects.create(
            article=new_article,
            embedding=embedding
        )
