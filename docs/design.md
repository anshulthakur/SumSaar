# Objectives

- At the end of the day/interval, fetch and collate news articles from different sources published during the defined interval.
- Group the articles into buckets, each bucket representing an overall NEWS item. The contents are articles that cover parts of, or the entire NEWS item.
- Weed out irrelevant/clickbait/lifestyle type articles
- Condense and summarize the content of the buckets into unified articles.


# UI considerations

- Offer the service in an app format as well as web-format. App format being the initial focus
- Users have profiles to store their preferences, and the recommendations and content quality adapt over time
- Feedback on each article:
  - Content quality
  - Content uniqueness/duplicates
  - Comment

# Fetching Items

## RSS Feeds
The first method, RSS Feeds. Almost all NEWS sources have RSS feeds. Some feeds will have the entire story as content in the feed, others syndicate only the titles and links, or truncated texts. This information is provided by the user while constructing the feed list.

# Comparing similarity
I could ask the LLM to compare the articles pairwise and bin them accordingly (that looks like an agent with tool-calling, or LLM-in-the-loop). Or, I could delve into the research on NLP etc and figure out an algorithmic way of doing that.

