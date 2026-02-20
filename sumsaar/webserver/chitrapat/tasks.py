from celery import shared_task
from sumsaar.rss import main as rss_main

@shared_task
def fetch_articles_task():
    """
    Celery task to fetch articles.
    """
    rss_main()
