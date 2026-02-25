from celery import shared_task
from sumsaar.rss import main as rss_main
from sumsaar.pipeline import process_incoming_article

@shared_task
def fetch_articles_task():
    """
    Celery task to fetch articles.
    """
    rss_main()

@shared_task
def process_item_task(raw_article_id):
    """
    Celery task to process a single raw article (Embed -> Search -> Synthesize).
    """
    process_incoming_article(raw_article_id)
