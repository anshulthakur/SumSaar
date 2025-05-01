import json
import sys

sys.path.append('/sumsaar/')

import pendulum

from airflow.sdk import dag, task
from datetime import timedelta

from sumsaar.rss import main
import logging


@dag(
    schedule="0 23 * * *",  # Everyday at 11
    start_date=pendulum.datetime(2025, 4, 30, tz="Asia/Kolkata"),
    catchup=False,
    tags=["sumsaar"],
    default_args={
        "depends_on_past": False,
        "retries": 0,
        "retry_delay": timedelta(minutes=5),
        # 'queue': 'bash_queue',
        # 'pool': 'backfill',
        # 'priority_weight': 10,
        # 'end_date': datetime(2016, 1, 1),
        # 'wait_for_downstream': False,
        # 'execution_timeout': timedelta(seconds=300),
        # 'on_failure_callback': some_function, # or list of functions
        # 'on_success_callback': some_other_function, # or list of functions
        # 'on_retry_callback': another_function, # or list of functions
        # 'sla_miss_callback': yet_another_function, # or list of functions
        # 'on_skipped_callback': another_function, #or list of functions
        # 'trigger_rule': 'all_success'
    },
)
def download_newspaper():
    @task()
    def download_newspaper(logical_date=None):
        """
        Download the day's RSS feeds
        """
        logger = logging.getLogger(__name__)
        #day = logical_date.date() + timedelta(days=1)# Logical date from the DAG's execution
        #print(day)

        #main(datefilter=day)
        main()

    download_newspaper()

download_newspaper()