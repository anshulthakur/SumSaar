You have built a robust, isolated ingestion pipeline. Now, we enter the most algorithmically complex part of SumSaar: Semantic Clustering.

This phase is exactly why you made the excellent decision to avoid relying entirely on LLMs. Using generative AI to compare hundreds of articles pairwise is slow, expensive, and prone to hallucinations. Phase 3 relies entirely on deterministic NLP and lightweight vector math to execute fast, accurate clustering while rigorously protecting your 8GB GPU limit.

Here is the definitive implementation guide for Phase 3: Worker 2 (CPU/GPU Bound).

---

### Step 1: Environment & Dependency Setup

This worker strictly handles PyTorch operations and traditional Natural Language Processing (NLP).

* Update your `requirements.txt` to include `torch`, `sentence-transformers`, `psycopg2-binary`, and `spacy`.
* Download a robust, offline NER (Named Entity Recognition) model for `spacy`, such as `en_core_web_sm` or `en_core_web_md`.
* Ensure this worker is strictly routed to the `ml_tasks` queue and deployed with Nvidia GPU runtime privileges.



### Step 2: The L2 Semantic Embedding Engine

The moment Worker 1 successfully stages a new article, it triggers `process_item(raw_article_id)`. Worker 2 immediately picks this up to convert the text into a mathematical representation.

* Load your chosen lightweight embedding model, such as `BAAI/bge-small-en-v1.5` or `sentence-transformers/all-MiniLM-L6-v2`.


* Initialize the model via PyTorch's `SentenceTransformer` and explicitly set the device to `cuda` to leverage your hardware.


* Strictly lower the default encoding batch size. The default batch size of 32 can cause sudden VRAM spikes on an 8GB GPU during high-volume ingestion.


* Pass the `text_content` of the `RawArticle` into the model to generate a 384-dimensional vector.



### Step 3: Fast Retrieval via pgvector

Instead of loading all active stories into memory, we delegate the heavy lifting directly to PostgreSQL.

* Connect to the unified PostgreSQL database.


* Execute a nearest-neighbor vector query targeting the `ArticleVector` table using `pgvector`.


* Constrain the search query using a time-decay window to only look at articles updated within the last 48 hours, preventing the system from comparing against years of archived news.


* Filter the database results to only return clusters where the cosine similarity score is greater than 0.80.


* If the database returns zero results above the 0.80 threshold, immediately branch to **Step 5 (New Story)**. If results are returned, proceed to Step 4.

### Step 4: L3 Event Verification (Anti-Collusion Layer)

This is the most critical safeguard in your entire architecture. A cosine similarity of 0.85 means the vocabulary is similar, but it does *not* guarantee they are the exact same event. This prevents the system from confusing two distinct building collapses (e.g., one in Delhi, one in UP) simply because they share words like "rubble," "casualties," and "rescue".

* Pass both the incoming `RawArticle` text and the matched `SynthesizedArticle` text through your `spacy` NER pipeline.


* Extract Named Entities specifically tagged as GPE (Locations), ORG (Organizations), PERSON, and EVENT.


* Extract temporal dimensions (Dates and Times).


* Perform a fast programmatic set intersection (Jaccard Similarity) on these extracted entities.


* If the primary locations or dates explicitly contradict each other, reject the L2 semantic match, classify the incoming article as a completely distinct event, and branch to **Step 5 (New Story)**.



### Step 5: Cluster Resolution & Interface Handoff

Once the relationship between the incoming article and the database is mathematically and factually confirmed, Worker 2 updates the state and hands off to the generative LLM.

**Scenario A: Match Confirmed (Update Story)**

* Create an association in the database linking the `raw_article_id` to the matched `StoryCluster`.


* Trigger the Interface 3 handoff by firing `update_story.delay(raw_article_id, cluster_id)`.


* Route this task specifically to the `llm_tasks` queue so it does not block subsequent incoming articles.



**Scenario B: No Match or Match Rejected (New Story)**

* Create a new row in the `StoryCluster` table and flag its status as `developing`.


* Create a new, blank row in the `SynthesizedArticle` table linked to this cluster.


* Save the newly generated 384-dimensional embedding into the `ArticleVector` table.


* Trigger the Interface 3 handoff by firing `update_story.delay(raw_article_id, new_cluster_id)`.



### Component Execution Summary

| Processing Stage | Technology | Core Objective |
| --- | --- | --- |
| **L2 Embedding** | PyTorch + bge-small | Transform raw text into 384-dim spatial vectors. |
| **Retrieval** | PostgreSQL (`pgvector`) | Rapid cosine distance matching ($> 0.80$) within a 48-hour window. |
| **L3 Verification** | spaCy (NER) | Extract and compare entities (Locations, Dates) to prevent false collusion. |
| **Delegation** | Celery | Route the validated UUIDs to the `llm_tasks` queue. |


## Graphical Interface

It is a great instinct to want to visualize Phase 3. Semantic clustering can quickly become an opaque "black box" if you aren't careful, and getting lost in the weeds of false positives is a rite of passage for data engineers. Because Phase 3 relies entirely on deterministic NLP and database queries rather than generative AI, it is highly quantifiable and perfect for visual debugging.

Based on your system architecture, the dashboard will visualize Phase 3 across two distinct views: a high-level systems monitor and a granular manual intervention workbench.

### 1. The Pipeline Health Dashboard (Macro View)

This section of your admin UI gives you a bird's-eye view of your hardware and database performance.

* 
**Queue & GPU Monitoring:** You will visualize the `ml_tasks` queue length in real-time, monitoring processing speeds to ensure your 8GB GPU is not bottlenecking the ingestion stream.


* 
**Vector Database Telemetry:** You will track the health of `pgvector`, visualizing the total number of active, developing `StoryClusters` and the total size of your `ArticleVector` table.



### 2. The Manual Intervention Workbench (Micro View)

This is where the magic of Phase 3 is exposed to your curators. Since the algorithm might occasionally make an error, you need a visual way to manually trigger a cluster split or merge.

* **The Similarity Matrix:** When viewing a `StoryCluster`, the UI will list the contributing `RawArticle` sources alongside their exact L2 Cosine Similarity scores.
* **L3 NER Debugger:** To help administrators understand *why* an article was grouped (or rejected), the dashboard will visually highlight the extracted Named Entities (Locations, Organizations, Dates) side-by-side using spaCy's visualizer style. This shows exactly how the L3 Event Verification algorithm prevented (or missed) a false collusion.

To truly understand why the visual L3 NER debugger is so critical for your curators, try adjusting the parameters in the sandbox below. Notice how relying purely on vector math (L2) can cause distinct events to falsely group together, and how L3 verification forces them apart.

### DB Queries

Tackling all the queries at once is the best approach. Because you are using Django as your backend, you have the choice between using the Django ORM (with the `pgvector` extension) or executing raw SQL for maximum performance on heavy analytics.

Here is the definitive guide to the underlying database queries that will power both the macro and micro views of your Phase 3 dashboard.

### 1. The Pipeline Health Dashboard (Macro Telemetry)

These queries populate the high-level charts and metrics on your admin landing page.

**A. Cluster Status Distribution**
To monitor the overall health of your clustering algorithm, you need to track the ratio of `developing` to `settled` stories. A sudden, massive spike in `developing` stories might indicate that your similarity threshold is too high, causing the system to spawn new clusters instead of updating existing ones.

* **Django ORM:**
```python
from django.db.models import Count
from your_app.models import StoryCluster

# Returns a dictionary like: {'developing': 142, 'settled': 890, 'archived': 5000}
cluster_stats = StoryCluster.objects.values('status').annotate(total=Count('status'))

```



**B. Vector Database Size & Performance**
As your 384-dimensional vector table grows, it will consume RAM. Monitoring the physical table size helps you anticipate when you might need to adjust your retention policies or add an HNSW (Hierarchical Navigable Small World) index to speed up nearest-neighbor searches.

* **Raw PostgreSQL (via Django `connection`):**
```sql
-- Returns the human-readable size of the vector table (e.g., "45 MB")
SELECT pg_size_pretty(pg_total_relation_size('your_app_articlevector'));

```



(Note: Queue monitoring for `ml_tasks` and `llm_tasks` is handled by Celery/Redis natively, usually visualized via the Flower tool, rather than PostgreSQL queries).

---

### 2. The Manual Intervention Workbench (Micro Debugging)

These are the exact queries that run when a curator clicks into a specific `StoryCluster` to debug why certain articles were grouped together.

**A. The Similarity Matrix (L2 Vector Query)**
When Worker 2 executes a semantic search, it looks for vectors with a cosine similarity greater than 0.80 , strictly within a 48-hour time-decay window to prevent comparing against old news. Here is the query that powers that matrix, calculating the exact mathematical distance.

* **Django ORM (using `pgvector`):**
```python
from pgvector.django import CosineDistance
from django.utils import timezone
from datetime import timedelta
from your_app.models import ArticleVector

time_threshold = timezone.now() - timedelta(hours=48)

# incoming_vector is the 384-dim array generated from the new article
matches = ArticleVector.objects.filter(
    article__cluster__last_updated__gte=time_threshold
).annotate(
    similarity=1 - CosineDistance('embedding', incoming_vector)
).filter(
    similarity__gt=0.80
).order_by('-similarity')

for match in matches:
    print(f"Cluster: {match.article.cluster.title}, Score: {match.similarity}")

```


* **Raw PostgreSQL Equivalent:**
```sql
SELECT 
    v.article_id, 
    1 - (v.embedding <=> '[... 384-dim vector ...]') AS cosine_similarity 
FROM 
    your_app_articlevector v
JOIN 
    your_app_synthesizedarticle s ON v.article_id = s.id
JOIN 
    your_app_storycluster c ON s.cluster_id = c.id
WHERE 
    1 - (v.embedding <=> '[... 384-dim vector ...]') > 0.80
    AND c.last_updated >= NOW() - INTERVAL '48 hours'
ORDER BY 
    cosine_similarity DESC;

```



**B. Fetching Sources for L3 NER Debugger**
When the dashboard renders the side-by-side L3 Named Entity visualizer, it needs to pull the exact raw text that contributed to the current cluster. According to your data contracts, the `SynthesizedArticle` maintains an array of contributing source URLs in a JSONB field.

* **Django ORM:**
```python
from your_app.models import SynthesizedArticle, RawArticle

# 1. Get the synthesized article for the cluster being reviewed
synth_article = SynthesizedArticle.objects.get(cluster_id=target_cluster_id)

# 2. Extract the list of source URLs from the JSONB array
contributing_urls = synth_article.sources 

# 3. Fetch the actual raw article payloads for the spaCy NER visualizer
raw_articles = RawArticle.objects.filter(url__in=contributing_urls)

for raw in raw_articles:
    # Pass raw.source_data['text_content'] into your spaCy pipeline
    pass

```