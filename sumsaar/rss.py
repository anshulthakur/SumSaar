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
from collections import defaultdict

rewritten_articles = []

progress = {'date': datetime.today(),
            'stage': None,
            'last_processed_index': [0,1]}

import signal, os

def signal_handler(signum, frame):
    global rewritten_articles
    global progress
    print('Stopping')
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
        self.system_prompt = ("You are an AI editor."
                              " Your task is to merge and rewrite the provided articles into a single cohesive, objective, and well-structured piece."
                            "Always generate objective and neutral articles without bias and only based on the information contained in the articles."
                        )
        self.template=("Combine the following articles into a single, well-structured news piece. "
                       "Ensure clarity, coherence, and eliminate redundant information. "
                       "It is important that you don't lose out on details, so be thorough."
                       "Maintain an objective tone.\n\n"
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

def fetch(max_entries=0):
    if progress['stage'] == None:
        progress['stage'] = 'fetch'
    elif progress['stage'] != 'fetch':
        #Already fetched for the day. Skip
        print('Skip fetching')
        return
    
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
    def generate(prompt):
        response = llm.client.generate( model='llama3.2:latest', 
                                                system=llm.system_prompt,
                                                prompt=llm.template.format(prompt=prompt),
                                                #template=llm.template,
                                                stream=False,
                                                keep_alive='1m',
                                                options={
                                                   'num_ctx': 8196*2,
                                                    'repeat_last_n': 0,
                                                    'temperature': 0.5,
                                                },
                                                format=LLMArticle.model_json_schema())
        return LLMArticle.model_validate_json(response.response)

    global rewritten_articles
    similarity_data = {}

    llm = CopyWriterLLM()

    with open(os.path.join(feeds_dir, 'compacted_similarity_ids.json'), 'r') as fd:
        similarity_data = json.load(fd)

    with open(os.path.join(feeds_dir, 'cache.json'), 'r') as fd:
        cache = json.load(fd)

    # Sort articles by ID and store (id, title, content)
    articles = { str(article['id']): article for article in cache}

    progress['stage'] = 'rewrite'

    init_outer_index = progress['last_processed_index'][0]
    init_inner_index = progress['last_processed_index'][1]

    for ii in range(init_outer_index, len(similarity_data)):
        article_group = similarity_data[ii]
        if init_outer_index == 0 and init_inner_index == 1:
            reference_content = {'title': articles[str(article_group[init_outer_index])]['title'],
                                'content': articles[str(article_group[init_outer_index])]['content'],
                                'urls': [articles[str(article_group[init_outer_index])]['link']]}
            rewritten_articles.append(reference_content)
        else:
            reference_content = {'title': rewritten_articles[-1]['title'],
                                'content': rewritten_articles[-1]['content'],
                                'urls': rewritten_articles[-1]['urls']}
        print(f'Article ID: {article_group[0]}')
        
        if len(article_group)>1:
            for jj in range(init_inner_index, len(article_group)):
                related_id = article_group[jj]
                #print('Reference:')
                print(reference_content)

                #print(f'Related: {related_id["id"]}')
                related_article = articles[str(related_id)]
                #print(related_article)
                prompt = f"Article 1: {reference_content['content']}\nArticle 2: {related_article['content']}"
                #llm.system_prompt = llm.system_prompt_template.format(reference_article = reference_content['content'])
                response_content = generate(prompt)
                while response_content.content == '...':
                    #Retry
                    response_content = generate(prompt)
                reference_content['title'] = response_content.title
                reference_content['content'] = response_content.content
                reference_content['urls'].append(related_article['link'])

                rewritten_articles[-1] = reference_content
                progress['last_processed_index'] = [ii,jj]

                print('Rewritten article:')
            print(reference_content)
    with open(os.path.join(feeds_dir, 'rewritten_articles.json'), "w") as json_file:
        json.dump(rewritten_articles, json_file, indent=4, default=str)


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
    with open(os.path.join(feeds_dir, 'similarity_results_combined.json'), 'r') as fd:
        similarity_data = json.load(fd)

    edges = []
    ids = []
    #edges = [(1, 2), (1, 3), (2, 4), (5, 6)]
    for article_id in similarity_data:
        if int(article_id) not in ids:
            ids.append(int(article_id))
        info = similarity_data[article_id]
        for related_id in info['scores']['LSA']['strong']:
            if int(article_id) < int(related_id['id']) and (int(article_id), int(related_id['id'])) not in edges:
                edges.append((int(article_id), int(related_id['id'])))
            elif int(article_id) > int(related_id['id']) and (int(related_id['id']), int(article_id)) not in edges:
                edges.append((int(related_id['id']), int(article_id)))
    forest = build_forest(edges, set(ids))

    with open(os.path.join(feeds_dir, 'compacted_similarity_ids.json'), 'w') as fd:
        json.dump(forest, fd, indent=2, default=str)

def load_progress():
    global progress
    global rewritten_articles
    try:
        with open(os.path.join(feeds_dir, 'progress.json'), 'r') as fd:
            progress = json.load(fd)
        with open(os.path.join(feeds_dir, 'rewritten_articles.json'), 'r') as fd:
            rewritten_articles = json.load(fd)
    except:
        traceback.print_exc()
        pass

    print(progress)

def save_progress():
    global progress
    global rewritten_articles
    with open(os.path.join(feeds_dir, 'rewritten_articles.json'), "w") as json_file:
        json.dump(rewritten_articles, json_file, indent=4, default=str)
    with open(os.path.join(feeds_dir, 'progress.json'), "w") as json_file:
        json.dump(progress, json_file, indent=4, default=str)

def clear_cache():
    global progress
    try:
        if os.path.exists(os.path.join(feeds_dir, 'cache.json')):
            os.remove(os.path.join(feeds_dir, 'cache.json'))
        if os.path.exists(os.path.join(feeds_dir, 'similarity_results_combined.json')):
            os.remove(os.path.join(feeds_dir, 'similarity_results_combined.json'))
        if os.path.exists(os.path.join(feeds_dir, 'compacted_similarity_ids.json')):
            os.remove(os.path.join(feeds_dir, 'compacted_similarity_ids.json'))
        if os.path.exists(os.path.join(feeds_dir, 'progress.json')):
            os.remove(os.path.join(feeds_dir, 'progress.json'))
    except:
        pass
    finally:
        progress = {'date': datetime.today(),
                    'stage': None,
                    'last_processed_index': [0,1]}

if __name__=="__main__":
    signal.signal(signal.SIGINT, signal_handler)
    load_progress() #2025-04-28 11:53:23.110309
    if datetime.strptime(progress['date'], "%Y-%m-%d %H:%M:%S.%f").date() != datetime.today().date():
        print('Clear cache')
        clear_cache()

    fetch(max_entries=0)
    if progress['stage'] == 'fetch':
        group_articles()
        progress['stage'] = 'grouped'
    print('Grouping done')
    if progress['stage'] != 'compacted':
        #Already grouped, skip
        compact()
        progress['stage'] = 'compacted'
    print('Compacting done')
    rewrite()
    save_progress()
    #dedup()
    #digest()
    
