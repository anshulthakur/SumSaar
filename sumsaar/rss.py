import feedparser
import os
from pydantic import BaseModel 
from datetime import datetime
from bs4 import BeautifulSoup

from settings import PROJECT_DIRS, OLLAMA_URL

from ollama import Client

feeds_dir = PROJECT_DIRS.get('runtime')

import feedparser
from datetime import datetime

import json
# Define the date format and filtering date
#filter_date = datetime(2025, 4, 5)  # Example filter date

class FeedItem(BaseModel):
    title: str
    link: str
    content: str
    date: str

def get_feed_list():
    feed_urls = []
    with open(os.path.join(feeds_dir, 'feeds.csv'), 'r') as fd:
        for line in fd:
            if len(line.strip())>0 and \
                not line.strip().startswith('#') and \
                line.strip() not in feed_urls:
                feed_urls.append(line.strip())
    return feed_urls

def parse_feed(url, datefilter= datetime.now().date(), max_entries=5):
    date_format = "%a, %d %b %Y %H:%M:%S %z"  # Adjust format based on feed
    feed_obj = feedparser.parse(url)
    entries = 0
    for entry in feed_obj.entries:
        if entries >= max_entries:
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
                feed = FeedItem(title= entry.title,
                                link= entry.link,
                                content=sanitized_content,
                                date=entry.published)
                entries += 1
                yield feed

def digest():
    client = Client(
            host=OLLAMA_URL,
            )
    
    # system_prompt = ("You are a seasoned and highly objective journalist and editor who ghostwrites one minute summaries from the provided news article. "
    #                 #"If you find that the current story is related to a previous story,"
    #                 #" you flag it for possible duplicates with the opening words [POSSIBLE DUPLICATE]. "
    #                 "If the article does not cover the 5Ws of journalism, you state it in the opening sentence of the article as [OPINION]."
    #                 "Further, if you find that the entire article is an opinion piece instead of reporting, you state it in the opening sentence of the article as [OPINION]. "
    #                 "IMPORTANT: Only provide the summary, and don't give any preface such as 'Here is a summary'. ")
    
    # system_prompt = ("Summarize the article provided by extracting the main facts, key events, and significant details" 
    #                  "related to current affairs, politics, economics, or global developments. "
    #                  "Ensure the summary remains concise, factual, and avoids subjective opinions "
    #                  "or lifestyle-related elements. " 
    #                  "If the article does not contain hard news but is more of a lifestyle, entertainment, " 
    #                  "or opinion piece, respond by noting that the content does not qualify as news.")

    system_prompt = ("Summarize the article provided by highlighting the key events and factual "
                     "details related to current affairs, politics, economics, or global developments. "
                     "Keep the summary concise and free from predefined structures or bullet points. "
                     "If the article does not contain hard news but is more of a lifestyle, entertainment, "
                     "or opinion piece, simply state: 'This is not a news article' without further explanation.")
    summaries = []
    feed_list = get_feed_list()
    for feed_url in feed_list:
        for feed_item in parse_feed(feed_url):
            response = client.chat(model='llama3.2:latest', messages=[
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': feed_item.content}
                        ],
                        stream=False,
                        keep_alive='1m',
                        options={
                            'num_ctx': 8196,
                            'repeat_last_n': 0,
                            'temperature': 0.5,
                        })
            summaries.append({'title': feed_item.title,
                            'link': feed_item.link,
                            'summary': response.message.content})
            print(response)

    with open(os.path.join(feeds_dir, 'summaries.json'), 'w') as fd:
        fd.write(json.dumps(summaries, indent=2))

if __name__=="__main__":
    digest()