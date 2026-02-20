import requests
import time
import os
from typing import Dict, Any
from sumsaar.settings import CRAWLER_URL

from playwright.sync_api import sync_playwright
import newspaper 
import logging

logger = logging.getLogger(__name__)

class Crawl4Ai:
    def __init__(self, base_url: str = "http://localhost:11235", api_token: str = None):
        self.base_url = base_url
        self.api_token = (
            api_token or os.getenv("CRAWL4AI_API_TOKEN") or "test_api_code"
        )  # Check environment variable as fallback
        self.headers = (
            {"Authorization": f"Bearer {self.api_token}"} if self.api_token else {}
        )

    def submit_and_wait(
        self, request_data: Dict[str, Any], timeout: int = 300
    ) -> Dict[str, Any]:
        # Submit crawl job
        response = requests.post(
            f"{self.base_url}/crawl", json=request_data, headers=self.headers
        )
        if response.status_code == 403:
            raise Exception("API token is invalid or missing")
        task_id = response.json()["task_id"]
        print(f"Task ID: {task_id}")

        # Poll for result
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Task {task_id} did not complete within {timeout} seconds"
                )

            result = requests.get(
                f"{self.base_url}/task/{task_id}", headers=self.headers
            )
            status = result.json()

            if status["status"] == "failed":
                print("Task failed:", status.get("error"))
                raise Exception(f"Task failed: {status.get('error')}")

            if status["status"] == "completed":
                return status

            time.sleep(2)

    def submit_sync(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/crawl_sync",
            json=request_data,
            headers=self.headers,
            timeout=60,
        )
        if response.status_code == 408:
            raise TimeoutError("Task did not complete within server timeout")
        response.raise_for_status()
        return response.json()

    def crawl_direct(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Directly crawl without using task queue"""
        response = requests.post(
            f"{self.base_url}/crawl_direct", json=request_data, headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
class Browser(object):
    pass


def scrape_with_playwright(url):
    # Using Playwright to render JavaScript
    content = ''
    with sync_playwright() as p:
        ws_endpoint = os.getenv("PLAYWRIGHT_WS_ENDPOINT")
        if ws_endpoint:
            logger.info(f'Trying remote endpoint {ws_endpoint}.')
            # Retry connection logic as the container might be starting up
            for attempt in range(5):
                try:
                    browser = p.chromium.connect(ws_endpoint, timeout=30000)
                    break
                except Exception as e:
                    logger.warning(f"Attempt {attempt+1}: Could not connect to Playwright at {ws_endpoint}: {e}")
                    time.sleep(2)
            else:
                logger.error("Failed to connect to Playwright after multiple attempts.")
                return None
        else:
            logger.info(f'Launch local')
            browser = p.chromium.launch()

        try:
            logger.info(f'Opening browser.')
            page = browser.new_page()
            logger.info(f'Opening {url}.')
            page.goto(url)
            #time.sleep(2) # Allow the javascript to render
            content = page.content()
        finally:
            browser.close()

    # Using Newspaper4k to parse the page content
    if len(content)>0:
        article = newspaper.article(url, input_html=content, language='en')

        return article
    else:
        return None