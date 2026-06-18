Below is the summarized, step-by-step implementation plan for SumSaar. It is structured to run efficiently within a VRAM constrained GPU while cleanly separating deterministic processing from LLM synthesis.

### Phase 1: Core Architecture & Database Setup

The foundation relies on an event-driven architecture that eschews daily batches in favor of continuous streaming.
* Initialize a Django project to serve as the web backend, API server, and user profile manager.
* Configure a unified PostgreSQL database as your single source of truth.
* Install the PostgreSQL `pgvector` extension to enable 384-dimensional semantic search and user profiling.
* Set up the `RawArticle` database model utilizing PostgreSQL's JSONB format to flexibly store transient data.
* Enforce database size management by implementing a strict 7-day hard-delete retention policy on raw items.
* Configure the `SynthesizedArticle` model to store the LLM-written narrative and maintain a structured list of atomic facts in a JSONB `facts_timeline` field.
* Set up Celery with a Redis broker to handle distributed pipeline queuing and asynchronous task execution.

### Phase 2: Worker 1 - The Ingestion Pipeline (I/O Bound)

This worker focuses entirely on pulling down data and filtering out exact duplicates before they waste compute resources.

* Use a scheduler like Celery Beat to trigger the `ingest_feed` task every 15 minutes.
* Fetch raw data across diverse sources, including RSS feeds, sitemaps, custom scraping lists, and social media APIs.
* Utilize headless browser rendering tools like Playwright or Puppeteer to extract content from Javascript-heavy websites.
* Throttle the Reddit API ingestion strictly to 58 requests per minute to prevent 429 Too Many Requests errors.
* Utilize managed APIs or authenticated developer webhooks for Twitter ingestion to bypass the severe instability of unofficial scrapers.
* Perform Level-1 exact deduplication by checking the database for incoming URLs that have already been crawled.
* Implement Locality-Sensitive Hashing (LSH) using MinHash to aggressively deduplicate heavily overlapping "near-duplicate" texts before saving.

### Phase 3: Worker 2 - Semantic Clustering (CPU/GPU)

This worker groups similar articles while strictly preventing false similarities using natural language processing.

* Trigger a `process_item` task the moment a new article successfully passes the staging layer.
* Generate a semantic embedding for the text using a lightweight, fast model such as `sentence-transformers/all-MiniLM-L6-v2` or `BAAI/bge-small-en-v1.5`.
* Run the embedding generation via PyTorch's `SentenceTransformer` with device set to `cuda`.
* Lower the default encoding batch size to prevent sudden VRAM spikes on your 8GB GPU during high-volume ingestion.
* Execute a nearest-neighbor vector search via `pgvector` targeting articles updated within the last 48 hours.
* Verify the semantic match (cosine similarity greater than 0.80) by executing Level-3 Event Verification.
* Extract Named Entities (people, locations, organizations) and temporal dimensions to definitively confirm both articles discuss the exact same event.

### Phase 4: Worker 3 - LLM Synthesis & Incremental Updates

This worker handles the generative aspect, strictly constrained to fit alongside your embeddings within 8GB of VRAM.

* Orchestrate the LLM calls using the official `openai` Python SDK with a custom base URL to hit your local runner.
* Load a small language model like Qwen 2.5 (3B) or Llama 3.2 (3B) using 4-bit quantization to shrink its memory footprint to roughly 2.5GB.
* Instruct the LLM to act as a deterministic FactExtractor to pull atomic, structured facts out of the raw text and format them in JSON.
* Utilize a structured output library alongside the SDK to guarantee the model strictly adheres to your required JSON schema.
* Mitigate context window explosion by passing only these newly extracted atomic facts to the StoryUpdater prompt alongside the existing narrative.
* Command the LLM to resolve conflicts chronologically, overwriting contradictory numbers or outdated facts seamlessly.
* Implement the SlotSum framework, asking the LLM to generate a narrative template with empty slots that are populated directly by verified JSON facts.
* Schedule a background quality-control task to execute a full, from-scratch re-synthesis of developing stories every 6 hours to clear linguistic artifacts.


### Phase 5: The Serving Layer & Personalization

This is where the user views the "Living Stories" and the platform learns their preferences.

* Deliver the finalized narratives through a sleek Django template MVP interface.
* Implement hybrid discovery, combining `pgvector` for conceptual semantic queries with `tsvector` for exact keyword matches.
* Store explicit user preferences, like followed topics or specific excluded tags, natively in a JSONB database field.
* Update the user's implicit reading preferences dynamically by mathematically modifying their `interest_vector`.
* Calculate this dynamic preference shift using an exponential moving average: 

$$V_{\text{new}} = (1 - \alpha) \cdot V_{\text{old}} + \alpha \cdot V_{\text{article}}$$
.

* Maintain a low learning rate in the equation to ensure the platform adapts to long-term interests without overreacting to individual clicks.


### System Stack Summary

| Component | Technology | Primary Function |
| --- | --- | --- |
| **Backend & ORM** | Python / Django | API routing, admin dashboard, user profiling, serving web views. |
| **Database** | PostgreSQL | Unified storage using JSONB for staging and `pgvector` for semantic search. |
| **Task Queue** | Celery + Redis | Asynchronous pipeline orchestration and worker management. |
| **Embeddings** | all-MiniLM-L6-v2 / bge-small | High-speed, low-VRAM vector generation for L2 clustering. |
| **Generative LLM** | Qwen 2.5 (3B) / Llama 3.2 (3B) | JSON fact extraction and narrative synthesis via the OpenAI API client. |