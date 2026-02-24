import feedparser
import os
import csv
from pydantic import BaseModel, Field, TypeAdapter
from typing import List, Optional, TypeAlias

from datetime import datetime
from bs4 import BeautifulSoup
import logging
import re
import json

logger = logging.getLogger(__name__)

logging.basicConfig(
level=logging.INFO,
format="%(asctime)s [%(levelname)s] %(message)s"
)

if __name__=="__main__":
    import sys
    sys.path.append('../')

from sumsaar.settings import PROJECT_DIRS, OLLAMA_URL, SUMMARY_LLM
from openai import OpenAI

import django
from django.conf import settings
if not settings.configured:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "sumsaar.webserver.webserver.settings")
    django.setup()

from chitrapat.models import Article as DbArticle, RawArticle, PipelineState, SimilarityResult

from sumsaar.crawler import scrape_with_playwright

feeds_dir = PROJECT_DIRS.get('runtime')

import feedparser
from datetime import datetime, timedelta

import itertools

import json
# Define the date format and filtering date
#filter_date = datetime(2025, 4, 5)  # Example filter date

from enum import Enum

from newspaper import Article
from dateutil import parser

import traceback

from sumsaar.pipeline import group_articles
from collections import defaultdict

progress = {'date': datetime.today(),
            'stage': None,
            'last_processed_index': [0,0]}

import signal, os

def signal_handler(signum, frame):
    global progress
    logger.info('Stopping')
    save_progress()
    exit(0)


#import requests
#from bs4 import BeautifulSoup
#import spacy

#nlp = spacy.load('en_core_web_sm')

# def extract_main_content(url):
#     response = requests.get(url)
#     soup = BeautifulSoup(response.content, 'html.parser')
#     paragraphs = soup.find_all('p')
#     content = ' '.join([para.get_text() for para in paragraphs])
#     doc = nlp(content)
#     return doc.text

class FeedType(str, Enum):
    story = 'story'
    title = 'title'
    summary = 'summary'

class FeedSource(BaseModel):
    link: str
    feed_type: FeedType = FeedType.story

class FeedItem(BaseModel):
    title: str
    link: str
    content: str
    date: str
    id: Optional[int]=Field(gt=0)
    db_id: Optional[int] = None # If set, this is an existing DB article

ArticleList: TypeAlias = list[FeedItem]
ArticleListModel = TypeAdapter(ArticleList)

class LLMSummary(BaseModel):
  keywords: List[str]
  title: str
  summary: str

class LLMArticle(BaseModel):
  title: str
  content: str
  keywords: List[str] = []

class UniqueStatus(str, Enum):
    related = 'RELATED'
    unrelated = 'UNRELATED'

class LLMDedup(BaseModel):
  result: UniqueStatus

def get_feed_list():
    feed_list = []
    blacklisted_feeds = []
    try:
        with open(os.path.join(feeds_dir, 'feeds_blacklist.csv'), 'r') as fd:
            reader = csv.DictReader(fd)
            blacklisted_feeds = [row['substring'].strip() for row in reader]
    except:
        traceback.print_exc()
        pass
    with open(os.path.join(feeds_dir, 'feeds.csv'), 'r') as fd:
        reader = csv.DictReader(fd)
        feed_list = [FeedSource.model_validate(row) for row in reader]

    #Prune the feedlist from blacklisted URLs
    pop_list = []
    for ii in range(0, len(feed_list)):
        feed = feed_list[ii]
        if any(map(feed.link.__contains__, blacklisted_feeds)):
            pop_list.append(ii)
    for ii in reversed(pop_list): #Reversed to preserve index numbers of remaining entries
        feed_list.pop(ii)

    return feed_list

def parse_date(pub_date_str):
    try:
        return parser.parse(pub_date_str)
    except Exception as e:
        logger.info(f"Error parsing date: {e}")
        return None

def parse_feed(feed, datefilter= datetime.now().date(), max_entries=10, index=None):
    date_format = "%a, %d %b %Y %H:%M:%S %z"  # Adjust format based on feed
    feed_obj = feedparser.parse(feed.link)
    entries = 0
    for entry in feed_obj.entries:
        if entry.title in index:
            #Already parsed this one before.
            logger.info(f'Skip {entry.link}')
            entries += 1
            continue
        if max_entries>0 and entries >= max_entries:
            break
        #logger.info(entry)
        if hasattr(entry, "published"):
            logger.info(f'Fetching: {entry.link}')
            published_datetime = parse_date(entry.published)
            if published_datetime.date() == datefilter:
                #logger.info(entry)
                #Extract title
                title = entry.title
                # Extracting and sanitizing HTML content
                raw_content = next(
                    (content['value'] for content in entry.get('content', []) if content['type'] == 'text/html'),
                    ""
                )
                sanitized_content = BeautifulSoup(raw_content, "html.parser").get_text()
                #logger.info(sanitized_content)
                if feed.feed_type != FeedType.story or len(sanitized_content)==0:
                    #Crawl the content out of the website using the link
                    try:
                        logger.info(f'Crawling for {entry.link}')
                        # request = {
                        #             "urls": entry.link,
                        #             "priority": 10,
                        #         }
                        # result = Crawl4Ai(base_url=CRAWLER_URL).submit_and_wait(request_data=request)
                        # sanitized_content = result["result"]["markdown"]
                        # sanitized_content = extract_main_content(entry.link)

                        article = Article(url=entry.link)
                        article.download()
                        article.parse()
                        sanitized_content = article.text
                        #logger.info(sanitized_content)
                    except:
                        logger.info(f'Error crawling news for {entry.link}. Try playwright')
                        #Try playwright crawler
                        article = scrape_with_playwright(url=entry.link)
                        if article is not None:
                            sanitized_content = article.text
                        else:
                            sanitized_content = ''
                    pass
                if len(sanitized_content)>0:
                    entries += 1
                    yield FeedItem(title= title,
                                    link= entry.link,
                                    content=sanitized_content,
                                    date=entry.published,
                                    id=None)
                    

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
        self.system_prompt_template = ''
        self.system_prompt = ''


class SummaryLLM(LLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.system_prompt = ("You are a seasoned and highly objective journalist and editor who ghostwrites one minute summaries from the provided news article. "
        #                 #"If you find that the current story is related to a previous story,"
        #                 #" you flag it for possible duplicates with the opening words [POSSIBLE DUPLICATE]. "
        #                 "If the article does not cover the 5Ws of journalism, you state it in the opening sentence of the article as [OPINION]."
        #                 "Further, if you find that the entire article is an opinion piece instead of reporting, you state it in the opening sentence of the article as [OPINION]. "
        #                 "IMPORTANT: Only provide the summary, and don't give any preface such as 'Here is a summary'. ")
        
        # self.system_prompt = ("Summarize the article provided by extracting the main facts, key events, and significant details" 
        #                  "related to current affairs, politics, economics, or global developments. "
        #                  "Ensure the summary remains concise, factual, and avoids subjective opinions "
        #                  "or lifestyle-related elements. " 
        #                  "If the article does not contain hard news but is more of a lifestyle, entertainment, " 
        #                  "or opinion piece, respond by noting that the content does not qualify as news.")

        self.system_prompt_template = ("Summarize the article provided by highlighting the key events and factual "
                        "details related to current affairs, politics, economics, or global developments. "
                        "Keep the summary concise and free from predefined structures or bullet points. "
                        "If the article does not contain hard news but is more of a lifestyle, entertainment, "
                        "or opinion piece, simply state: 'This is not a news article' without further explanation."
                        "IMPORTANT: Always return the output in JSON format with structure: {\"title\": <title of article>, \"content\": <summary>, \"keywords\": <list of salient keywords in article>}")
        self.system_prompt = self.system_prompt_template

class CopyWriterLLM(LLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.system_prompt = ("You are an AI editor."
        #                       " Your task is to merge and rewrite the provided articles into a single cohesive, objective, and well-structured piece."
        #                     "Always generate objective and neutral articles without bias and only based on the information contained in the articles."
        #                     "IMPORTANT: Always return the output in JSON format with structure: {\"title\": <title of article>, \"content\": <summary>, \"keywords\": <list of salient keywords in article>}"
        #                 )
        
        self.system_prompt = ("You are a summarization and synthesis assistant. I will provide two articles."
                              "Your task is to: Write a merged synthesis article:"
                                "- Integrate the information into a coherent narrative, weaving together both sources."
                                "- Use structured sections or bullet points as needed."
                                "- Do not omit nuance or subtle points."
                                "- Maintain balance and factual accuracy."
                                "- Avoid introducing information not present in the sources."
                                "- Use a professional, neutral tone suitable for publication."
                                "- Organize with clear paragraphs and logical flow."
                                "IMPORTANT: Always return the output in JSON format with structure: {\"title\": <title of article>, \"content\": <summary>, \"keywords\": <list of salient keywords in article>}"
                            )
        self.template=("Combine the following articles into a single, well-structured news piece. "
                       "Ensure clarity, coherence, and eliminate redundant information."
                       "It is important that you don't lose out on details, so be consise and thorough.\n\n"
                       "The content of the article must be well defined markdown, but do not add any ``` tags."
                       "Source Articles:\n{prompt}\n\n"
                       )


class DedupLLM(LLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.system_prompt_template = ("You are an AI language model tasked with assessing the similarity between the reference article provided below and the one user provides."
                              " Based on their topic, key events, and overall subject matter, "
                              "classify their relationship using only one of the following fixed categories:"
                              "- **RELATED** if the articles discuss the same event, or subject"
                              "- **UNRELATED** if the articles cover entirely different events, or subjects"
                              "Your response must be **EXACTLY** one word, choosing **ONLY** from the given categories:"
                              " `RELATED` or `UNRELATED`. No explanations, no additional text—just the classification. "
                              "The reference article is : {reference_article}"
                        )

def fetch_context_articles(days=3):
    """Fetches recent articles from DB to provide context for deduplication."""
    cutoff = datetime.now() - timedelta(days=days)
    # We use the Django ORM to fetch recent articles
    db_articles = DbArticle.objects.filter(updated_at__gte=cutoff)
    items = []
    for art in db_articles:
        items.append(FeedItem(
            title=art.title,
            link=art.source_urls[0] if art.source_urls else f"http://db/{art.id}",
            content=art.content,
            date=str(art.updated_at),
            id=None, # Will be assigned in fetch
            db_id=art.id
        ))
    return items

def fetch(max_entries=0, datefilter=datetime.now().date()):
    if progress['stage'] == None or progress['stage']=='finished':
        progress['stage'] = 'fetch'
    elif progress['stage'] != 'fetch':
        #Already fetched for the day. Skip
        logger.info('Skip fetching')
        return
    
    feed_list = get_feed_list()
    ii = 0
    articles = []
    
    # Load existing articles from MongoDB
    raw_articles = RawArticle.objects.all().order_by('feed_id')
    for ra in raw_articles:
        articles.append(FeedItem(
            title=ra.title,
            link=ra.link,
            content=ra.content,
            date=str(ra.published_date) if ra.published_date else "",
            id=ra.feed_id,
            db_id=ra.db_id
        ))

    crawled_titles = []
    for article in articles:
        ii = max(ii, int(article.id))
        # If we are resuming, we might have DB items in cache; track them
        if article.db_id:
            crawled_titles.append(article.title)

    # Load context from DB (articles from last 3 days)
    # This ensures we compare new fetches against recent history
    if ii == 0: # Only load context if we are starting fresh or cache is empty
        context_articles = fetch_context_articles(days=3)
        for ctx_item in context_articles:
            ii += 1
            ctx_item.id = ii
            articles.append(ctx_item)
            crawled_titles.append(ctx_item.title)

    for article in articles:
        # Recalculate max ID in case we added context
        if article.id:
            ii = max(ii, int(article.id))
        crawled_titles.append(article.title)

    for feed in feed_list:
        try:
            for feed_item in parse_feed(feed, max_entries=max_entries, index=crawled_titles, datefilter= datefilter):
                ii += 1
                feed_item.id = ii
                articles.append(feed_item)
                
                # Save to MongoDB
                RawArticle.objects.create(
                    feed_id=feed_item.id,
                    title=feed_item.title,
                    link=feed_item.link,
                    content=feed_item.content,
                    published_date=parse_date(feed_item.date) if feed_item.date else None,
                    db_id=feed_item.db_id
                )
        except:
            logger.info('Exception occurred')
            traceback.print_exc()

    

def digest():
    llm = SummaryLLM(host=OLLAMA_URL)
    
    summaries = []
    with open(os.path.join(feeds_dir, 'cache.json'), 'r') as fd:
        articles = json.load(fd)
    ii = 0
    for article in articles:
        response = llm.client.chat.completions.create(model=SUMMARY_LLM, messages=[
                        {'role': 'system', 'content': llm.system_prompt},
                        {'role': 'user', 'content': article.content}
                    ],
                    temperature=0.5,
                    extra_body={
                        'keep_alive': '1m',
                        'options': {
                            'num_ctx': 8196,
                            'repeat_last_n': 0,
                        },
                        #'format': LLMSummary.model_json_schema()
                    },
                    )
        summary = LLMSummary.model_validate_json(response.choices[0].message.content)
        ii += 1
        summaries.append({'id': ii,
                            'title': article.title,
                            'link': article.link,
                            'content': article.content,
                            'summary': {
                                'title': summary.title,
                                'keywords': summary.keywords,
                                'content':summary.summary}
                        })
        logger.info(summary)

    with open(os.path.join(feeds_dir, 'summaries.json'), 'w') as fd:
        fd.write(json.dumps(summaries, indent=2))


def dedup():
    summaries = []
    with open(os.path.join(feeds_dir, 'cache.json'), 'r') as fd:
        summaries = json.load(fd)
    llm = DedupLLM()
    # Iterate over articles pairwise without repetition
    for ref_article, compare_article in itertools.combinations(summaries, 2):
        llm.system_prompt = llm.system_prompt_template.format(reference_article = ref_article['content'])
        response = llm.client.chat.completions.create(model=SUMMARY_LLM, messages=[
                            {'role': 'system', 'content': llm.system_prompt},
                            {'role': 'user', 'content': compare_article['content']}
                        ],
                        temperature=0.5,
                        extra_body={
                            'keep_alive': '1m',
                            'options': {
                                'num_ctx': 8196,
                                'repeat_last_n': 0,
                            },
                            'format': LLMDedup.model_json_schema()
                        }
                        )
        related = LLMDedup.model_validate_json(response.choices[0].message.content)
        if (related.result == UniqueStatus.related):
            logger.info(f"{compare_article['title']} seems related to {ref_article['title']}")
        else:
            logger.info(f"{compare_article['title']} is UNRELATED to {ref_article['title']}")

def rewrite():
    def clean_control_chars(s: str) -> str:
        # Remove ASCII control characters except \t, \n, \r if you want to keep them
        return re.sub(r'[\x00-\x1F]', '', s)

    def generate(prompt):
        try:
            response = llm.client.chat.completions.create( model=SUMMARY_LLM, 
                                            messages=[
                                                {'role': 'system', 'content': llm.system_prompt},
                                                {'role': 'user', 'content': llm.template.format(prompt=prompt)}
                                            ],
                                            temperature=0.3,
                                            top_p = 0.8,
                                            seed = 0,
                                            response_format = LLMArticle.model_json_schema(),
                                            extra_body={
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
                                            })
            print(response)
            return LLMArticle.model_validate_json(clean_control_chars(response.choices[0].message.content))
        except:
            logger.error('LLM timedout.')
            response = LLMArticle(title='', content='...')
            raise
            return response

    similarity_data = {}

    llm = CopyWriterLLM()

    # Load clusters from PipelineState
    state = PipelineState.objects.first()
    similarity_data = state.clusters if state else []

    progress['stage'] = 'rewrite'

    if len(progress['last_processed_index'])==0:
        progress['last_processed_index'] = [0,0]
    init_outer_index = progress['last_processed_index'][0]
    init_inner_index = progress['last_processed_index'][1]

    for ii in range(init_outer_index, len(similarity_data)):
        article_group = similarity_data[ii]
        
        logger.info(f'Article ID: {article_group[0]}, Number of articles: {len(article_group)}')

        # Get reference article
        try:
            first_raw = RawArticle.objects.get(feed_id=article_group[0])
        except RawArticle.DoesNotExist:
            logger.error(f"RawArticle {article_group[0]} not found")
            continue

        db_article = None
        if first_raw.db_id:
            try:
                db_article = DbArticle.objects.get(id=first_raw.db_id)
            except DbArticle.DoesNotExist:
                pass
        
        if not db_article:
            db_article = DbArticle.objects.create(
                title=first_raw.title,
                content=first_raw.content,
                source_urls=[first_raw.link]
            )
            first_raw.db_id = db_article.id
            first_raw.save()
            logger.info(f'Created DbArticle {db_article.id}')
        else:
            logger.info(f'Using existing DbArticle {db_article.id}')

        start_jj = 1
        if ii == init_outer_index:
            start_jj = max(1, init_inner_index + 1)

        if len(article_group) > 1:
            for jj in range(start_jj, len(article_group)):
                related_id = article_group[jj]
                try:
                    related_raw = RawArticle.objects.get(feed_id=related_id)
                except RawArticle.DoesNotExist:
                    continue
                
                # Check if already merged
                if related_raw.link in db_article.source_urls:
                    logger.info(f"Skipping {related_id}, already merged.")
                    continue

                logger.info(f'Merging {related_id} into {db_article.id}')
                prompt = f"Article 1: {db_article.content}\nArticle 2: {related_raw.content}"
                
                try:
                    response_content = generate(prompt)
                    while response_content.content == '...':
                        #Retry
                        response_content = generate(prompt)
                    
                    db_article.title = response_content.title
                    db_article.content = response_content.content
                    db_article.keywords = response_content.keywords
                    if related_raw.link not in db_article.source_urls:
                        db_article.source_urls.append(related_raw.link)
                    db_article.save()
                    
                    if not related_raw.db_id:
                        related_raw.db_id = db_article.id
                        related_raw.save()

                    progress['last_processed_index'] = [ii,jj]
                    save_progress()
                    logger.info('Rewritten article updated')
                except Exception as e:
                    logger.error(f"Error merging articles: {e}")
                    raise

        progress['last_processed_index'] = [ii + 1, 0]
        save_progress()

    progress['stage'] = 'finished'
    progress['last_processed_index'] = [0, 0]
    save_progress()


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, node):
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])  # Path compression
        return self.parent[node]

    def union(self, node1, node2):
        root1 = self.find(node1)
        root2 = self.find(node2)
        if root1 != root2:
            self.parent[root2] = root1  # Merge components

    def add(self, node):
        if node not in self.parent:
            self.parent[node] = node

    def get_components(self):
        components = {}
        for node in self.parent:
            root = self.find(node)
            if root not in components:
                components[root] = set()
            components[root].add(node)
        return [sorted(component) for component in components.values()]  # Sorting each subgraph

def build_forest(edges, all_ids):
    uf = UnionFind()

    # Step 1: Add all nodes
    for node in all_ids:
        uf.add(node)

    # Step 2: Merge connected components
    for src, dest in edges:
        uf.union(src, dest)

    # Step 3: Extract and sort connected components
    return uf.get_components()

def compact():
    # Load similarity results from DB
    sim_results = SimilarityResult.objects.all()
    similarity_data = {str(res.reference_id): {'scores': res.scores} for res in sim_results}

    edges = []
    ids = []
    #edges = [(1, 2), (1, 3), (2, 4), (5, 6)]
    for article_id in similarity_data:
        if int(article_id) not in ids:
            ids.append(int(article_id))
        info = similarity_data[article_id]
        # Check if MiniLM scores exist
        minilm_scores = info['scores'].get('MiniLM', {}).get('strong', [])
        for related_id in minilm_scores:
            if int(article_id) < int(related_id['id']) and (int(article_id), int(related_id['id'])) not in edges:
                edges.append((int(article_id), int(related_id['id'])))
            elif int(article_id) > int(related_id['id']) and (int(related_id['id']), int(article_id)) not in edges:
                edges.append((int(related_id['id']), int(article_id)))
    forest = build_forest(edges, set(ids))

    # Save forest to PipelineState
    state = PipelineState.objects.first()
    if not state:
        state = PipelineState()
    state.clusters = forest
    state.save()

def load_progress():
    global progress
    try:
        state = PipelineState.objects.first()
        if state:
            progress['stage'] = state.stage
            progress['last_processed_index'] = state.last_processed_index
    except:
        traceback.print_exc()
        pass

    logger.info(progress)

def save_progress():
    global progress
    
    state = PipelineState.objects.first()
    if not state:
        state = PipelineState()
    state.stage = progress['stage']
    state.last_processed_index = progress['last_processed_index']
    state.save()

def clear_cache():
    global progress
    try:
        RawArticle.objects.all().delete()
        SimilarityResult.objects.all().delete()
        PipelineState.objects.all().delete()
    except:
        pass
    finally:
        progress = {'date': datetime.today(),
                    'stage': None,
                    'last_processed_index': [0,0]}


def main(datefilter=datetime.now().date()):
    load_progress() #2025-04-28 11:53:23.110309
    if progress['date'].date() != datetime.today().date():
        logger.info('Clear cache')
        clear_cache()

    fetch(max_entries=20, datefilter=datefilter)
    if progress['stage'] == 'fetch':
        group_articles()
        progress['stage'] = 'grouped'
    logger.info('Grouping done')
    if progress['stage'] != 'compacted':
        #Already grouped, skip
        compact()
        progress['stage'] = 'compacted'
    logger.info('Compacting done')
    try:
        rewrite()
    except:
        traceback.print_exc()
    finally:
        save_progress()
    #dedup()
    #digest()

if __name__=="__main__":
    #signal.signal(signal.SIGINT, signal_handler)
    main()
    
