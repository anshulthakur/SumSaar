It is great that we nailed down the interfaces first. Because we established the "Pass IDs, not data" rule and strict JSONB schemas, you can build Phase 2 (The Ingestion Pipeline) in a complete vacuum. You do not need to worry about the LLM, the GPU, or vector math yet. Your only goal here is to fetch dirty data, clean it, ruthlessly filter out duplicates, and save it to the database.

Scraping the modern web is notoriously flaky. Sites block crawlers, JavaScript fails to load, and APIs change. This phase embraces that chaos by failing gracefully and ensuring bad data never crosses the interface boundary into your machine learning queue.

Here is the detailed implementation guide for Phase 2: Worker 1 (I/O Bound).

---

### Step 1: Environment & Dependency Setup

Worker 1 requires specific libraries dedicated to I/O operations and text hashing.

* Update your `requirements.txt` to include `newspaper3k` (or `newspaper4k`), `playwright`, `feedparser`, `praw` (for Reddit), and `datasketch` (for MinHash LSH).
* Run `playwright install` inside your Docker container or local environment to download the required headless browser binaries.

---

### Step 2: The Celery Beat Scheduler

We move away from daily batches to a continuous stream. The scheduler acts as the heartbeat of your ingestion pipeline.

* Configure Celery Beat in your `celery.py` file to trigger the master orchestration task.
* Set a crontab schedule to fire `dispatch_feeds()` every 15 minutes.
* Write the `dispatch_feeds()` function to query your database for all active news sources (e.g., RSS URLs, Subreddits) and push individual `ingest_source.delay(source_url, source_type)` messages to the `io_tasks` Redis queue.



---

### Step 3: The Fetching Engine

This is the core of `Worker 1`. It routes the incoming task based on the `source_type` parameter.

**RSS & Sitemaps (`source_type: 'rss'`)**

* Use `feedparser` to extract all recent article URLs from the RSS XML feed.
* For each URL, iterate through the parsing logic.

**Standard Web Articles (`source_type: 'web'`)**

* Attempt to extract the article using `newspaper4k` first, as it is fast and lightweight.


* Implement a `try/except` fallback: if `newspaper4k` returns empty text (indicating a JavaScript-heavy site), spin up a headless Playwright instance to render the DOM and extract the text natively.



**Reddit (`source_type: 'reddit'`)**

* Configure the PRAW client using OAuth credentials.
* Implement a strict rate-limiting wrapper. Reddit enforces a 60 request/minute limit; throttle your worker to exactly 58 requests per minute to maintain a safety buffer.
* Implement exponential backoff to handle HTTP 429 (Too Many Requests) gracefully without crashing the worker.



**Twitter (`source_type: 'twitter'`)**

* Avoid using open-source scrapers like Nitter, as they suffer from extreme instability.
* Integrate a managed API (like Apify) or authenticated developer webhooks to fetch tweets securely and reliably.

---

### Step 4: The Deduplication Layer (L1 & LSH)

Before touching the database, you must aggressively filter out redundancies to save your 8GB GPU from unnecessary embedding computations.

* **L1 Deduplication (Exact Match):** Execute a fast PostgreSQL query to check if the incoming URL already exists in the `RawArticle` table. Discard immediately if found.


* **Near-Duplicate Filtering (LSH):** Syndicated news often changes only the headline or the first paragraph. Initialize a MinHash object from the `datasketch` library.


* Tokenize the cleaned article text and feed it into the MinHash algorithm.
* Compare the resulting hash against a local cache (or Redis) of hashes from the last 48 hours. If the Jaccard similarity is above 0.85, classify it as a near-duplicate and discard it.



---

### Step 5: Database Commit & Interface Handoff

Once an article passes extraction and deduplication, it must be staged according to the strict JSONB contract.

* Construct the dictionary ensuring it strictly contains the keys: `title`, `text_content`, `author`, and `published_date`.


* Save the record to the `RawArticle` PostgreSQL table using the Django ORM.


* Wrap the database save in an atomic transaction to ensure data integrity.
* Upon successful commit, extract the newly generated UUID.
* Trigger the Interface 2 handoff by calling `process_item.delay(raw_article.id)` and routing it specifically to the `ml_tasks` queue.



### Error Handling Protocol

| Failure Type | Action Taken | Queue Status |
| --- | --- | --- |
| **HTTP 404 / 403** | Log error, discard task. | Cleared |
| **Timeout / Network Drop** | Retry task up to 3 times with exponential backoff. | Re-queued |
| **Missing JSONB Keys** | Django validator rejects save, log data malformation. | Cleared (Prevents poisoning) |
| **Rate Limit Hit (429)** | Pause worker execution for 60 seconds. | Re-queued |

## GUI for visualization and verification
Building a dedicated space for manual ground-truthing is an incredibly smart move. You cannot trust an LLM's synthesis blindly, especially in a news context where factual accuracy is paramount. Because you have already chosen Django as your web backend, you have the perfect toolkit to build an administrative layer that provides both pipeline visibility and manual review capabilities.

Here is the detailed implementation plan for incorporating the Pipeline Health Dashboard and the Ground-Truthing UI into your architecture.

### 1. The Pipeline Health Dashboard

To fulfill the requirement of monitoring worker queues, processing speeds, and crawler success rates, you do not need to build a task tracking system from scratch.

* Install **Flower**, an open-source, real-time web-based monitor built specifically for Celery.
* Add Flower as a dedicated service in your `docker-compose.yml` file, pointing it to your Redis broker.
* Use the Flower UI to visually track the queue lengths for your distinct `io_tasks`, `ml_tasks`, and `llm_tasks` queues.
* Monitor task failure rates (e.g., Reddit API 429 errors or Playwright timeouts) directly from this interface to gauge pipeline health.
* Embed a link to the Flower dashboard directly in the navigation bar of your custom Django admin interface.

### 2. The Ground-Truthing Interface

This custom Django view serves as the workbench for administrators to verify the AI's output against the source material.

* Create a dedicated Django view (e.g., `ClusterVerificationView`) restricted via the `@staff_member_required` decorator.
* Design a split-screen UI: the left panel displays the `SynthesizedArticle`, and the right panel displays a list of the contributing `RawArticle` sources.
* Render the `SynthesizedArticle` content inside an interactive Markdown editor to allow admins to manually rewrite or edit the synthesized narrative.


* Include buttons to manually trigger a cluster split or merge if the L3 Event Verification algorithm made an error.


* Include a "Test Bench" section in the UI to allow administrators to tweak the LLM prompt templates and run one-off generation tests.



### 3. Safe Local HTML Rendering

Since your `RawArticle` model stores raw, transient data flexibly in a PostgreSQL JSONB field, you already have the payload needed to reconstruct the original web page. However, rendering scraped HTML directly in your dashboard is a massive security risk (Cross-Site Scripting).

* Ensure that your scraping worker specifically saves the raw HTML payload into a key like `source_data['raw_html']` alongside the parsed title and author.


* Create a specific Django URL endpoint (e.g., `/admin/article/<uuid>/render/`) dedicated solely to serving this raw payload.
* Write the corresponding view to extract the HTML string from the JSONB field and return it as a pure `HttpResponse`.
* Strip out any tracking pixels or injected ad-scripts from the HTML string before returning the response using a lightweight library like `bleach` (if strict text-only rendering is preferred).
* In your Ground-Truthing UI, embed an `<iframe>` element to display the source article.
* Set the `src` of the iframe to your local render endpoint and strictly apply the `sandbox` attribute (e.g., `sandbox="allow-same-origin"`). This guarantees the visual layout of the news site is preserved for your review while completely neutralizing any malicious JavaScript hiding in the scraped DOM.