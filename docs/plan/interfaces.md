The Golden Rule for your event-driven interfaces is this: **Pass IDs, not data.** Redis is a message broker, not a database. Your Celery workers must pass lightweight UUIDs through the message queue and use PostgreSQL as the single source of truth to fetch the heavy textual payloads.

Here are the definitive interfaces, data formats, and access methods between your components.

---

### Interface 1: The Ingestion Contract (Scheduler ➔ Worker 1)

This interface governs how the system initiates data collection. It is strictly I/O-bound.

* **Method of Access:** Celery Beat scheduler pushes a task to the `io_tasks` Redis queue.
* **Task Signature:** `ingest_source(source_url: str, source_type: str)`
* **Input Payload (Redis):** ```json
{
"source_url": "https://example-news.com/rss",
"source_type": "rss"
}
```
* **Output Contract (PostgreSQL):** Worker 1 must cleanly parse the raw HTML or RSS XML, perform L1 exact-match deduplication , and insert a new row into the `RawArticle` table.
* **Trigger:** Upon a successful database commit, Worker 1 immediately fires `process_item.delay(raw_article_id)`.

---

### Interface 2: The Clustering Contract (Worker 1 ➔ Worker 2)

This interface bridges the gap between raw data collection and semantic processing. It binds specifically to the GPU for vector operations.

* **Method of Access:** Worker 1 pushes a task to the `ml_tasks` Redis queue.
* **Task Signature:** `process_item(raw_article_id: UUID)`
* **Input Payload (Redis):**
```json
{
  "raw_article_id": "a1b2c3d4-e5f6-7890-1234-56789abcdef0"
}

```

* **Data Access (PostgreSQL):** Worker 2 uses the Django ORM to query the `RawArticle` row. The contract demands that the `source_data` JSONB field conforms strictly to this schema:


```json
{
  "title": "Raw scraped headline",
  "text_content": "Cleaned markdown text of the article...",
  "author": "Jane Doe",
  "published_date": "2026-06-16T10:00:00Z"
}

```


* **Output Contract (PostgreSQL):** Worker 2 generates a 384-dimensional embedding and executes a pgvector search.

* *If New:* Creates a new `StoryCluster` and a new `SynthesizedArticle`.
* *If Match:* Associates the `raw_article_id` with an existing `StoryCluster`.




* **Trigger:** Worker 2 fires `update_story.delay(raw_article_id, cluster_id)`.

---

### Interface 3: The LLM Synthesis Contract (Worker 2 ➔ Worker 3)

This is where deterministic clustering hands off to the generative LLM to update the "Living Story."

* **Method of Access:** Worker 2 pushes a task to the `ml_tasks` (or a dedicated `llm_tasks`) Redis queue.
* **Task Signature:** `update_story(raw_article_id: UUID, cluster_id: UUID)`
* **Input Payload (Redis):**
```json
{
  "raw_article_id": "a1b2c3d4-...",
  "cluster_id": "f9e8d7c6-..."
}

```


* **Data Access (PostgreSQL):** Worker 3 fetches the new `RawArticle` text and the existing `SynthesizedArticle` linked to the `cluster_id`.


* **Intermediate Data Format (The FactExtractor):** Worker 3 prompts the LLM to output a strict JSON list of atomic facts. This is your protection against hallucination and context window explosion.


```json
{
  "extracted_facts": [
    {"entity": "Casualties", "value": "15", "timestamp": "2026-06-16T12:00:00Z"},
    {"entity": "Location", "value": "New Delhi", "timestamp": "2026-06-16T12:00:00Z"}
  ]
}

```


* **Output Contract (PostgreSQL):** Worker 3 must append these new facts to the `facts_timeline` JSONB field. It then overwrites the `content` text field of the `SynthesizedArticle` with the newly integrated narrative and updates the `sources` array with the new URL.



---

### Summary of Solid Interface Boundaries

| Component Boundary | Transport Mechanism | Payload Passed | State Retrieved From |
| --- | --- | --- | --- |
| **Scheduler ➔ I/O** | Redis (`io_tasks`) | Target URL / ID | N/A (External Network) |
| **I/O ➔ Clustering** | Redis (`ml_tasks`) | `raw_article_id` | `RawArticle` (PostgreSQL) |
| **Clustering ➔ LLM** | Redis (`ml_tasks`) | `raw_article_id`, `cluster_id` | `RawArticle`, `StoryCluster` (PostgreSQL) |

By treating the Django ORM models as your strict data contracts and Celery as a lightweight signaling mechanism, you can now build Worker 1 in complete isolation. You only need to mock the `process_item.delay()` call to verify Worker 1 is doing its job perfectly.