import feedparser
import os
import csv
from pydantic import BaseModel, Field, TypeAdapter
from typing import List, Optional, TypeAlias

from datetime import datetime
from bs4 import BeautifulSoup

from settings import PROJECT_DIRS, OLLAMA_URL
from ollama import Client

from crawler import scrape_with_playwright

feeds_dir = PROJECT_DIRS.get('runtime')

import feedparser
from datetime import datetime

import itertools

import json
# Define the date format and filtering date
#filter_date = datetime(2025, 4, 5)  # Example filter date

from enum import Enum

from newspaper import Article
from dateutil import parser

import traceback

from pipeline import group_articles

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

ArticleList: TypeAlias = list[FeedItem]
ArticleListModel = TypeAdapter(ArticleList)

class LLMSummary(BaseModel):
  keywords: List[str]
  title: str
  summary: str

class LLMArticle(BaseModel):
  title: str
  content: str

class UniqueStatus(str, Enum):
    related = 'RELATED'
    unrelated = 'UNRELATED'

class LLMDedup(BaseModel):
  result: UniqueStatus

def get_feed_list():
    feed_list = []
    with open(os.path.join(feeds_dir, 'feeds.csv'), 'r') as fd:
        reader = csv.DictReader(fd)
        feed_list = [FeedSource.model_validate(row) for row in reader]
    return feed_list

def parse_date(pub_date_str):
    try:
        return parser.parse(pub_date_str)
    except Exception as e:
        print(f"Error parsing date: {e}")
        return None

def parse_feed(feed, datefilter= datetime.now().date(), max_entries=10, index=None):
    date_format = "%a, %d %b %Y %H:%M:%S %z"  # Adjust format based on feed
    feed_obj = feedparser.parse(feed.link)
    entries = 0
    for entry in feed_obj.entries:
        if entry.title in index:
            #Already parsed this one before.
            print(f'Skip {entry.link}')
            entries += 1
            continue
        if max_entries>0 and entries >= max_entries:
            break
        #print(entry)
        if hasattr(entry, "published"):
            print(f'Fetching: {entry.link}')
            published_datetime = parse_date(entry.published)
            if published_datetime.date() == datefilter:
                #print(entry)
                #Extract title
                title = entry.title
                # Extracting and sanitizing HTML content
                raw_content = next(
                    (content['value'] for content in entry.get('content', []) if content['type'] == 'text/html'),
                    ""
                )
                sanitized_content = BeautifulSoup(raw_content, "html.parser").get_text()
                #print(sanitized_content)
                if feed.feed_type != FeedType.story or len(sanitized_content)==0:
                    #Crawl the content out of the website using the link
                    try:
                        print(f'Crawling for {entry.link}')
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
                        #print(sanitized_content)
                    except:
                        print(f'Error crawling news for {entry.link}. Try playwright')
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
        self.client = Client(
            host=OLLAMA_URL,
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
                        "or opinion piece, simply state: 'This is not a news article' without further explanation.")
        self.system_prompt = self.system_prompt_template

class CopyWriterLLM(LLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.system_prompt_template = ("Rewrite the following news article while retaining all the relevant details "
                        "and adding further details from the article provided by the user."
                        "Keep the tone of the article objective and neutral."
                        "Minimize redundancy of information and limit the article to a maximum of 1000 words."
                        "The reference article is : {reference_article}")


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

def fetch(max_entries=0):
    feed_list = get_feed_list()
    ii = 0
    articles = []
    try:
        with open(os.path.join(feeds_dir, 'cache.json'), 'r') as fd:
            articles_json = json.load(fd)
            articles = ArticleListModel.validate_json(json.dumps(articles_json))
    except:
        pass
    crawled_titles = []
    for article in articles:
        ii = max(ii, int(article.id))
        crawled_titles.append(article.title)

    for feed in feed_list:
        try:
            for feed_item in parse_feed(feed, max_entries=max_entries, index=crawled_titles):
                ii += 1
                feed_item.id = ii
                articles.append(feed_item)
        except:
            print('Exception occurred')
            traceback.print_exc()

    with open(os.path.join(feeds_dir, 'cache.json'), 'wb') as fd:
        #print(type(ArticleListModel.dump_json(articles, indent=2)))
        fd.write(ArticleListModel.dump_json(articles, indent=2))
        #fd.write(json.dumps(articles.json(), indent=2))

def digest():
    llm = SummaryLLM(host=OLLAMA_URL)
    
    summaries = []
    with open(os.path.join(feeds_dir, 'cache.json'), 'r') as fd:
        articles = json.load(fd)
    ii = 0
    for article in articles:
        response = llm.client.chat(model='deepseek-r1:14b', messages=[
                        {'role': 'system', 'content': llm.system_prompt},
                        {'role': 'user', 'content': article.content}
                    ],
                    stream=False,
                    keep_alive='1m',
                    options={
                        'num_ctx': 8196,
                        'repeat_last_n': 0,
                        'temperature': 0.5,
                    },
                    format=LLMSummary.model_json_schema())
        summary = LLMSummary.model_validate_json(response.message.content)
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
        print(summary)

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
        response = llm.client.chat(model='deepseek-r1:14b', messages=[
                            {'role': 'system', 'content': llm.system_prompt},
                            {'role': 'user', 'content': compare_article['content']}
                        ],
                        stream=False,
                        keep_alive='1m',
                        options={
                            'num_ctx': 8196,
                            'repeat_last_n': 0,
                            'temperature': 0.5,
                        },
                        format=LLMDedup.model_json_schema())
        related = LLMDedup.model_validate_json(response.message.content)
        if (related.result == UniqueStatus.related):
            print(f"{compare_article['title']} seems related to {ref_article['title']}")
        else:
            print(f"{compare_article['title']} is UNRELATED to {ref_article['title']}")

def rewrite():
    rewritten_articles = []
    similarity_data = {}
    covered_ids = []
    llm = CopyWriterLLM()

    with open(os.path.join(feeds_dir, 'similarity_results_combined.json'), 'r') as fd:
        similarity_data = json.load(fd)

    with open(os.path.join(feeds_dir, 'cache.json'), 'r') as fd:
        cache = json.load(fd)

    # Sort articles by ID and store (id, title, content)
    articles = { str(article['id']): article for article in cache}

    for article_id in similarity_data:
        if article_id in covered_ids:
            #Skip article as it is already covered
            print(f'Skip {article_id}')
            continue
        print(f'Article ID: {article_id}')
        info = similarity_data[article_id]
        
        reference_content = {'title': '',
                             'content': articles[str(article_id)]}
        
        if len(info['scores']['LSA']['strong'])>0:
            for related_id in info['scores']['LSA']['strong']:
                #print('Reference:')
                print(reference_content)

                #print(f'Related: {related_id["id"]}')
                related_article = articles[str(related_id['id'])]
                #print(related_article)
                llm.system_prompt = llm.system_prompt_template.format(reference_article = reference_content['content'])
                response = llm.client.chat(model='deepseek-r1:14b', messages=[
                                    {'role': 'system', 'content': llm.system_prompt},
                                    {'role': 'user', 'content': related_article['content']}
                                ],
                                stream=False,
                                keep_alive='1m',
                                options={
                                    'num_ctx': 8196*2,
                                    'repeat_last_n': 0,
                                    'temperature': 0.5,
                                },
                                format=LLMArticle.model_json_schema())
                response_content = LLMArticle.model_validate_json(response.message.content)
                reference_content = {'title': response_content.title,
                                     'content': response_content.content}
                print('Rewritten article:')
                if related_id['id'] not in covered_ids:
                    covered_ids.append(related_id['id'])
        rewritten_articles.append(reference_content)
        if article_id not in covered_ids:
            covered_ids.append(article_id)
    with open(os.path.join(feeds_dir, 'rewritten_articles.json'), "w") as json_file:
        json.dump(rewritten_articles, json_file, indent=4, default=str)
    
if __name__=="__main__":
    #fetch(max_entries=0)
    #group_articles()
    rewrite()
    #dedup()
    #digest()
    