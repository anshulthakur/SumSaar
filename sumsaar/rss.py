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

from chitrapat.models import RawArticle
from chitrapat.tasks import process_item_task

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
                    

def fetch(max_entries=0, datefilter=datetime.now().date()):
    feed_list = get_feed_list()
    ii = 0
    articles = []
    
    crawled_titles = []
    
    # Check against existing URLs in RawArticle (Staging)
    existing_links = set(RawArticle.objects.values_list('url', flat=True))

    for feed in feed_list:
        try:
            for feed_item in parse_feed(feed, max_entries=max_entries, index=crawled_titles, datefilter= datefilter):
                if feed_item.link in existing_links:
                    logger.info(f"Skipping existing link: {feed_item.link}")
                    continue

                # Save to PostgreSQL (Staging)
                ra = RawArticle.objects.create(
                    url=feed_item.link,
                    source_data={
                        'title': feed_item.title,
                        'content': feed_item.content,
                        'date': str(feed_item.date) if feed_item.date else None,
                        'feed_id': ii # Legacy tracking, optional
                    }
                )
                existing_links.add(feed_item.link)
                
                # Trigger the processing pipeline immediately
                logger.info(f"Triggering processing for {ra.id}")
                process_item_task.delay(ra.id)

        except:
            logger.info('Exception occurred')
            traceback.print_exc()

    

def main(datefilter=datetime.now().date()):
    fetch(max_entries=20, datefilter=datefilter)

if __name__=="__main__":
    main()
    
