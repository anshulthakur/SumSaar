import feedparser
import os
import csv
from pydantic import BaseModel
from typing import List

from datetime import datetime
from bs4 import BeautifulSoup
from crawler import Crawl4Ai

from settings import PROJECT_DIRS, OLLAMA_URL, CRAWLER_URL
from ollama import Client

feeds_dir = PROJECT_DIRS.get('runtime')

import feedparser
from datetime import datetime

import itertools

import json
# Define the date format and filtering date
#filter_date = datetime(2025, 4, 5)  # Example filter date

from enum import Enum

from newspaper import Article

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

class LLMSummary(BaseModel):
  keywords: List[str]
  title: str
  summary: str

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

def parse_feed(feed, datefilter= datetime.now().date(), max_entries=10):
    date_format = "%a, %d %b %Y %H:%M:%S %z"  # Adjust format based on feed
    feed_obj = feedparser.parse(feed.link)
    entries = 0
    for entry in feed_obj.entries:
        if max_entries>0 and entries >= max_entries:
            break
        #print(entry)
        if hasattr(entry, "published"):
            published_datetime = datetime.strptime(entry.published, date_format)
            if published_datetime.date() == datefilter:
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
                        print(f'Try to crawl news for {entry.link}')
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
                        print(f'Error crawling news for {entry.link}')
                        sanitized_content = ''
                    pass
                if len(sanitized_content)>0:
                    entries += 1
                    yield FeedItem(title= entry.title,
                                    link= entry.link,
                                    content=sanitized_content,
                                    date=entry.published)
                    

class LLM(object):
    def __init__(self, host="http://localhost:11434", **kwargs):
        self.client = Client(
            host=OLLAMA_URL,
            )
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

        self.system_prompt = ("Summarize the article provided by highlighting the key events and factual "
                        "details related to current affairs, politics, economics, or global developments. "
                        "Keep the summary concise and free from predefined structures or bullet points. "
                        "If the article does not contain hard news but is more of a lifestyle, entertainment, "
                        "or opinion piece, simply state: 'This is not a news article' without further explanation.")
    

class DedupLLM(LLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.system_prompt = ("You are an AI language model tasked with assessing the similarity between the reference article provided below and the one user provides."
                              " Based on their topic, key events, and overall subject matter, "
                              "classify their relationship using only one of the following fixed categories:"
                              "- **RELATED** if the articles discuss the same event, or subject"
                              "- **UNRELATED** if the articles cover entirely different events, or subjects"
                              "Your response must be **EXACTLY** one word, choosing **ONLY** from the given categories:"
                              " `RELATED` or `UNRELATED`. No explanations, no additional text—just the classification. "
                              "The reference article is : {reference_article}"
                        )

def digest():
    llm = SummaryLLM(host=OLLAMA_URL)
    
    summaries = []
    feed_list = get_feed_list()
    ii = 0
    for feed in feed_list:
        for feed_item in parse_feed(feed, max_entries=0):
            response = llm.client.chat(model='deepseek-r1:14b', messages=[
                            {'role': 'system', 'content': llm.system_prompt},
                            {'role': 'user', 'content': feed_item.content}
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
                              'title': feed_item.title,
                              'link': feed_item.link,
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
    with open(os.path.join(feeds_dir, 'summaries.json'), 'r') as fd:
        summaries = json.load(fd)
    llm = DedupLLM()
    # Iterate over articles pairwise without repetition
    for ref_article, compare_article in itertools.combinations(summaries, 2):
        llm.system_prompt = llm.system_prompt.format(reference_article = ref_article['summary']['content'])
        response = llm.client.chat(model='deepseek-r1:14b', messages=[
                            {'role': 'system', 'content': llm.system_prompt},
                            {'role': 'user', 'content': compare_article['summary']['content']}
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

if __name__=="__main__":
    digest()
    dedup()