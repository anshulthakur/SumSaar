# SumSaar Architecture & Design Document

## 1. System Overview
**SumSaar** is an intelligent news aggregation and synthesis engine. Unlike traditional aggregators that simply list links, SumSaar ingests raw content from diverse sources (RSS, scraped web pages), groups them by semantic similarity, and uses LLMs to synthesize a single, evolving "Living Story" for each event.

The system is designed for **incremental processing**, ensuring that breaking news is published immediately while subsequent updates are seamlessly merged into the narrative without redundancy.

---

## 2. Data Architecture (Unified PostgreSQL)
We move away from a hybrid MongoDB/Postgres setup to a **Unified PostgreSQL** architecture. This leverages PostgreSQL's `JSONB` for flexible raw data storage and `pgvector` for semantic search, ensuring ACID compliance and simplified operations.

### 2.1 Core Models

#### **A. Staging Layer (Transient)**
*   **`RawArticle`**: Stores the raw, unprocessed feed items.
    *   `url` (Unique Index): To prevent re-ingestion.
    *   `source_data` (JSONB): Stores title, raw HTML, author, published date.
    *   `fetched_at`: Timestamp for retention policy.
    *   **Retention Policy**: Rows deleted after **7 days**.

#### **B. Synthesis Layer (Persistent)**
*   **`StoryCluster`**: Represents a distinct news event (e.g., "Earthquake in Japan").
    *   `title`: Auto-generated topic label.
    *   `status`: `developing`, `settled`, `archived`.
    *   `last_updated`: Timestamp of the last added fact.

*   **`SynthesizedArticle`**: The user-facing content.
    *   `cluster` (ForeignKey): Links to `StoryCluster`.
    *   `headline`: The current version of the headline.
    *   `content`: The LLM-written narrative.
    *   `facts_timeline` (JSONB): A structured list of atomic facts extracted so far (used for re-synthesis).
    *   `sources` (JSONB): List of `RawArticle` URLs contributing to this story.

*   **`ArticleVector`**: Semantic index.
    *   `article` (OneToOne): Links to `SynthesizedArticle`.
    *   `embedding` (VectorField): 384-dim vector (using `all-MiniLM-L6-v2`).

#### **C. User Layer**
*   **`UserProfile`**:
    *   `followed_topics` (JSONB): Explicit interests (e.g., ["AI", "Cricket"]).
    *   `interest_vector` (VectorField): A moving average vector of articles the user has engaged with.

*   **`UserInteraction`**:
    *   `user`, `article`, `interaction_type` (click, like, bookmark, time_spent).

---

## 3. The Pipeline (Incremental Hybrid Synthesis)

The pipeline operates as a continuous stream rather than a daily batch job.

### 3.1 Ingestion (Celery Task: `ingest_feed`)
1.  **Fetch**: Download RSS feed/Sitemap.
2.  **Dedup (L1)**: Check `RawArticle` table for existing URLs. Discard duplicates.
3.  **Store**: Save new items to `RawArticle`.
4.  **Trigger**: Spawn `process_item` task for each new item.

### 3.2 Processing (Celery Task: `process_item`)
1.  **Embed**: Generate a temporary vector for the incoming `RawArticle`.
2.  **Vector Search**: Query `ArticleVector` for nearest neighbors (Cosine Similarity > 0.80).
    *   *Filter*: Only search articles updated in the last 48 hours (unless it's a major ongoing saga).
3.  **Decision Logic**:
    *   **No Match**: Treat as a **New Story**.
        *   Create `StoryCluster` and `SynthesizedArticle`.
        *   Save embedding to `ArticleVector`.
    *   **Match Found**: Treat as an **Update**.
        *   Trigger `update_story` task with the matched `SynthesizedArticle`.

### 3.3 Synthesis (The "Hybrid" Engine)

#### **Scenario A: New Story**
*   **Action**: Direct summarization.
*   **Prompt**: "Write a news report based on this raw input."

#### **Scenario B: Incremental Update (Fast)**
*   **Step 1: Fact Extraction**:
    *   **Input**: New `RawArticle`.
    *   **LLM Role**: `FactExtractor`.
    *   **Output**: List of atomic facts (JSON).
*   **Step 2: Narrative Update**:
    *   **Input**: Current `SynthesizedArticle` + Extracted Facts.
    *   **LLM Role**: `StoryUpdater`.
    *   **Prompt**: "Update the story with these new facts. Overwrite outdated info (e.g., casualty numbers). Keep the tone neutral."

#### **Scenario C: Full Re-synthesis (Periodic/Quality Control)**
*   **Trigger**: Every 6 hours for "developing" clusters, or when `facts_timeline` grows by >20%.
*   **Action**: Re-write the entire article from scratch using the accumulated `facts_timeline` to ensure coherence and remove artifacts of incremental edits.

---

## 4. LLM Roles & Prompts

### 4.1 Fact Extractor
*   **Goal**: Compress raw text into pure signal.
*   **System Prompt**:
    > "Identify and extract the key facts, figures, dates, and distinct events from the text. Present them as a concise JSON list. Focus on hard information (who, what, when, where, why, numbers). Ignore opinions and fluff."

### 4.2 Story Updater
*   **Goal**: Maintain a living document.
*   **System Prompt**:
    > "You are a news editor maintaining a developing story. I will provide the 'Current Story' and a list of 'New Facts'. Update the story to incorporate the new facts. If a new fact contradicts old info, overwrite it. Do not delete relevant context."

### 4.3 Topic Modeler (Offline/Batch)
*   **Goal**: Assign readable tags to clusters.
*   **Method**: Zero-shot classification or simple keyword extraction on the `SynthesizedArticle` title/summary.

---

## 5. Search & Discovery

### 5.1 Hybrid Search
The search bar will support two modes simultaneously:
1.  **Semantic Search**: Uses `pgvector` to find conceptually similar articles (e.g., query "climate change" matches articles about "global warming" or "carbon tax").
2.  **Keyword Search**: Uses PostgreSQL `tsvector` (Full Text Search) for exact matches on names or specific entities.

### 5.2 Topic Modeling
*   Articles are auto-tagged with keywords during synthesis.
*   **Cluster Labeling**: A background job groups `StoryCluster`s into broader categories (e.g., "Politics", "Tech") using a predefined taxonomy and vector similarity.

---

## 6. User Personalization & Feedback

### 6.1 Explicit Preferences
*   Users can "Follow" specific entities (e.g., "SpaceX") or broad topics (e.g., "Technology").
*   **Implementation**: Store tags in `UserProfile.followed_topics`. Filter feed queries using `JSONB` containment (`@>`).

### 6.2 Implicit Learning (The "For You" Feed)
*   **Mechanism**:
    1.  When a user reads an article, fetch its `ArticleVector`.
    2.  Update the user's `interest_vector` using a weighted moving average:
        $$ V_{new} = (1 - \alpha) \cdot V_{old} + \alpha \cdot V_{article} $$
        *(Where $\alpha$ is the learning rate, e.g., 0.1)*
    3.  **Feed Generation**: Sort articles by Cosine Similarity between `ArticleVector` and `UserProfile.interest_vector`.

### 6.3 Feedback Loop
*   **Actions**:
    *   **Click**: Small positive reinforcement.
    *   **Like**: Strong positive reinforcement.
    *   **"Show Less Like This"**: Negative reinforcement (subtract vector or add to exclusion list).

---

## 7. Code Flow Sequence

1.  **Scheduler**: Airflow/Celery Beat triggers `fetch_feeds` every 15 mins.
2.  **Worker 1 (IO)**:
    *   Fetches RSS XML.
    *   Parses & cleans.
    *   Saves to `RawArticle`.
3.  **Worker 2 (CPU/GPU)**:
    *   Picks up `RawArticle`.
    *   Computes Embedding (MiniLM).
    *   Queries Postgres (`SELECT ... ORDER BY embedding <=> query_vec LIMIT 1`).
4.  **Worker 3 (LLM)**:
    *   **If Match**: Calls `FactExtractor` -> `StoryUpdater`. Updates `SynthesizedArticle`.
    *   **If No Match**: Calls `Summarizer`. Creates `SynthesizedArticle`.
5.  **Web Server (Django)**:
    *   Serves content.
    *   Captures `UserInteraction`.
    *   Updates `UserProfile` vector asynchronously.

---

## 8. Retention & Cleanup Policy
*   **Raw Data**: `RawArticle` rows are hard-deleted after **7 days**. This keeps the DB size manageable while allowing a window for duplicate detection.
*   **Vector Data**: `ArticleVector` rows persist as long as the `SynthesizedArticle` exists.
*   **Archival**: `SynthesizedArticle`s older than 1 year can be moved to "Cold Storage" (a separate archive table) to keep the active index fast.
```
