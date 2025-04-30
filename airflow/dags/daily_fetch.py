import json
import sys

sys.path.append('/sumsaar/')

import pendulum

from airflow.decorators import dag, task
from datetime import timedelta

from sumsaar.rss import main


@dag(
    schedule="0 23 * * 1-7",  # Weekdays at or after 6 PM
    start_date=pendulum.datetime(2025, 5, 30, tz="Asia/Kolkata"),
    catchup=True,
    tags=["sumsaar"],
)
def download_newspaper():
    @task()
    def download_newspaper(execution_date=None):
        """
        Download the day's RSS feeds
        """
        bulk_download = False
        day = execution_date.date() + timedelta(days=1)# Logical date from the DAG's execution

        main(datefilter=day)

    download_newspaper()


download_newspaper()