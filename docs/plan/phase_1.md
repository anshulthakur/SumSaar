Getting the foundation right is the most critical part of SumSaar. Phase 1 establishes the single source of truth and the orchestration mechanics that will allow your processing pipelines to run smoothly alongside your Django backend.

Here is the deep dive into Phase 1, detailing the microservices, the database schema, configurable options, and the Docker environment needed to bring it to life.

### 1. Microservices Architecture

To ensure your system scales and manages the 8GB GPU constraint effectively, SumSaar should be containerized into distinct services. This prevents I/O-bound scraping tasks from blocking your CPU/GPU-bound machine learning tasks.

* **`web` (Django):** The core API server, admin dashboard, and web interface.
* **`db` (PostgreSQL + pgvector):** The unified database handling relational data, JSONB staging, and vector indexing.
* **`redis`:** The message broker for Celery and in-memory cache for fast rate-limiting (crucial for the Reddit API).
* **`worker-io` (Celery):** Dedicated exclusively to fetching feeds, scraping web pages (via Playwright), and basic exact-match deduplication.
* **`worker-ml` (Celery):** Dedicated to semantic clustering (Embeddings) and LLM Orchestration. This worker will specifically bind to your Nvidia GPU.
* **`beat` (Celery):** The scheduler that triggers ingestion and re-synthesis tasks at defined intervals.

---

### 2. Database Models (Constituents)

These Django models represent the core schema, leveraging PostgreSQL's specific features.

#### A. The Staging Layer (Transient)

**`RawArticle`** 

* `id`: Primary Key (UUID).
* `url`: URLField (Unique Index to prevent duplicate ingestion).
* `source_data`: JSONField (Stores raw HTML, parsed markdown, title, author, publish date).
* `source_type`: CharField (e.g., 'rss', 'reddit', 'twitter', 'custom_scrape').
* `fetched_at`: DateTimeField (Auto-now-add. Used by the background cleanup task to enforce the 7-day hard-delete retention policy).

#### B. The Synthesis Layer (Persistent)

**`StoryCluster`** 

* `id`: Primary Key (UUID).
* `title`: CharField (Auto-generated topic label).
* `status`: CharField (Choices: `developing`, `settled`, `archived`).
* `last_updated`: DateTimeField.

**`SynthesizedArticle`** 

* `id`: Primary Key (UUID).
* `cluster`: ForeignKey (Linked to `StoryCluster`).
* `headline`: CharField.
* `content`: TextField (The LLM-written narrative).
* `facts_timeline`: JSONField (Structured list of atomic facts extracted by the LLM).
* `sources`: JSONField (Array of URLs contributing to this story).

**`ArticleVector`** 

* `article`: OneToOneField (Linked to `SynthesizedArticle`).
* `embedding`: VectorField (Using pgvector, specifically set to 384 dimensions to match `all-MiniLM-L6-v2` or `bge-small-en-v1.5`).


#### C. The User Layer

**`UserProfile`** 

* `user`: OneToOneField (Linked to Django's standard User model).
* `followed_topics`: JSONField (Explicit interests, e.g., `["AI", "Cricket"]`).
* `excluded_topics`: JSONField (Explicit blacklists).
* `interest_vector`: VectorField (384 dimensions, updated via implicit mathematical learning).

**`UserInteraction`**

* `user`: ForeignKey.
* `article`: ForeignKey (`SynthesizedArticle`).
* `interaction_type`: CharField (Choices: `click`, `like`, `bookmark`, `time_spent`).

---

### 3. Configurable Options

To maintain flexibility without touching code, specific variables should be exposed.

#### Developer Configurations (via `.env`)

* **LLM Routing:** `OPENAI_BASE_URL` and `OPENAI_API_KEY`. This allows you to hot-swap between your local Ollama instance and a cloud provider seamlessly.
* **Pipeline Throttling:** `REDDIT_RATE_LIMIT_PER_MIN` (Strictly default to 58).
* **Hardware Tuning:** `EMBEDDING_BATCH_SIZE`. Lower this value to prevent VRAM spikes on your 8GB GPU during high-volume ingestion.
* **Data Retention:** `RAW_ARTICLE_RETENTION_DAYS` (Default: 7).

#### User Configurations (via UI/Database)

* **Topic Filtering:** Include/Exclude keyword arrays stored in their JSONB profile.
* **Feed Customization:** Allowing users to submit specific RSS feeds to the global ingestion pipeline (subject to admin approval or isolated to their profile).

---

### 4. Docker Compose Blueprint

Below is the definitive `docker-compose.yml` to orchestrate Phase 1. It utilizes a specialized PostgreSQL image that comes pre-packaged with the `pgvector` extension.

```yaml
version: '3.8'

services:
  db:
    image: ankane/pgvector:latest # PostgreSQL with pgvector pre-installed
    container_name: sumsaar_db
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-sumsaar}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-postgres}
      POSTGRES_DB: ${POSTGRES_DB:-sumsaar_db}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    container_name: sumsaar_redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  web:
    build: .
    container_name: sumsaar_web
    command: python manage.py runserver 0.0.0.0:8000
    volumes:
      - .:/app
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgres://${POSTGRES_USER:-sumsaar}:${POSTGRES_PASSWORD:-postgres}@db:5432/${POSTGRES_DB:-sumsaar_db}
      - CELERY_BROKER_URL=redis://redis:6379/0
      - OPENAI_BASE_URL=${OPENAI_BASE_URL:-http://host.docker.internal:11434/v1} # Points to local Ollama by default
    depends_on:
      - db
      - redis

  worker-io:
    build: .
    container_name: sumsaar_worker_io
    command: celery -A sumsaar worker -l info -Q io_tasks --concurrency=4
    volumes:
      - .:/app
    environment:
      - DATABASE_URL=postgres://${POSTGRES_USER:-sumsaar}:${POSTGRES_PASSWORD:-postgres}@db:5432/${POSTGRES_DB:-sumsaar_db}
      - CELERY_BROKER_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis

  worker-ml:
    build: .
    container_name: sumsaar_worker_ml
    command: celery -A sumsaar worker -l info -Q ml_tasks --concurrency=1
    volumes:
      - .:/app
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - DATABASE_URL=postgres://${POSTGRES_USER:-sumsaar}:${POSTGRES_PASSWORD:-postgres}@db:5432/${POSTGRES_DB:-sumsaar_db}
      - CELERY_BROKER_URL=redis://redis:6379/0
      - OPENAI_BASE_URL=${OPENAI_BASE_URL:-http://host.docker.internal:11434/v1}
    depends_on:
      - db
      - redis

  beat:
    build: .
    container_name: sumsaar_beat
    command: celery -A sumsaar beat -l info
    volumes:
      - .:/app
    environment:
      - DATABASE_URL=postgres://${POSTGRES_USER:-sumsaar}:${POSTGRES_PASSWORD:-postgres}@db:5432/${POSTGRES_DB:-sumsaar_db}
      - CELERY_BROKER_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis

volumes:
  postgres_data:
  redis_data:

```

### 5. Stepwise Execution for Phase 1

1. **Initialize the Repository:** Set up the standard Python environment. Create a `requirements.txt` including `Django`, `psycopg2-binary`, `celery`, `redis`, `pgvector`, and `openai`.
2. **Spin Up Infrastructure:** Run `docker-compose up -d db redis` to establish the data layer.
3. **Configure Django:** Run `django-admin startproject sumsaar .` and modify `settings.py` to connect to PostgreSQL and Redis. Add `'pgvector.django'` to your `INSTALLED_APPS`.
4. **Define Models & Migrate:** Write the models defined in Section 2 into `models.py`. Run `python manage.py makemigrations` and `python manage.py migrate` to create the tables and the `vector` extension in PostgreSQL.
5. **Configure Celery:** Create a `celery.py` file next to `settings.py`. Define your task queues (`io_tasks` for fetching, `ml_tasks` for embeddings/LLM calls) to route tasks to the correct Docker containers.


## Refinements:

We do need to make a few **minor, high-value refinements** to the Phase 1 implementation details to enforce the rules of interfaces at the code level.

Here is how the finalized interfaces shape the finer details of Phase 1:

### 1. Enforcing the JSONB Contracts in Django

In Phase 1, we defined `source_data` in the `RawArticle` model and `facts_timeline` in the `SynthesizedArticle` model as generic `JSONField`s. To respect the SOLID interfaces we just established, we should enforce the schemas at the database level so a rogue web scraper cannot break the LLM pipeline.

* **Refinement:** Add a custom Django validator to the `RawArticle` model that explicitly checks if the incoming dictionary contains the exact keys: `title`, `text_content`, `author`, and `published_date`.
* **Why:** If Worker 1 (I/O) scrapes a badly formatted site and misses the `text_content` key, the Django ORM will reject the database save immediately. This prevents the system from passing a "poisoned" UUID to Worker 2, cleanly isolating the failure to the scraping layer.

### 2. Expanding the Celery Queue Routing

Initially, Phase 1 specified two queues: `io_tasks` and `ml_tasks`. Looking at the Interface 3 contract, Worker 2 (Embeddings) hands off to Worker 3 (LLM Synthesis).

* **Refinement:** In your `settings.py` / `celery.py` configuration, explicitly define three distinct routing queues: `io_tasks`, `ml_tasks` (for fast PyTorch vector generation), and `llm_tasks` (for slower OpenAI/Ollama API calls).
* **Why:** Even if `ml_tasks` and `llm_tasks` both execute on the same 8GB GPU machine, putting the LLM calls in their own queue prevents long-running generative tasks from blocking the rapid clustering of incoming news items.

### 3. Celery Task Chaining (The Triggers)

The interfaces dictate that one worker triggers the next upon a successful PostgreSQL commit.

* **Refinement:** When building the worker scripts in Phase 1, you will use Celery's `.delay()` method for simple hand-offs (e.g., `process_item.delay(raw_article.id)`). For more complex logic, like the 6-hour full re-synthesis job, you will leverage Celery's `chain` or `chord` primitives.
* **Why:** This ensures that the message to Worker 2 is only dispatched *after* the PostgreSQL transaction for Worker 1 has been fully committed, preventing race conditions where Worker 2 tries to read a UUID that doesn't exist in the database yet.

### 4. Docker Compose Updates

The `docker-compose.yml` blueprint provided in Phase 1 remains 95% intact, but it should be slightly updated to reflect the queue split.

* **Refinement:** You can split the `worker-ml` container definition into two logical services in the docker file: one listening to `-Q ml_tasks` and another listening to `-Q llm_tasks`.
* **Why:** This gives you the ability to independently scale or restart the LLM orchestration worker without interrupting the vector embedding worker.

### The Verdict

By defining the interfaces *first*, you have successfully pressure-tested the Phase 1 architecture. The Django ORM models serve as the strict data schemas, PostgreSQL acts as the stateful blackboard, and Redis + Celery act as the stateless signaling network. You are completely clear to begin writing the code for Phase 1.