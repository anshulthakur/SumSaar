# SumSaar - Project Context for Gemini

## 1. Objective
The goal of **SumSaar** is to build a news aggregation pipeline that:
1.  Ingests news stories from multiple RSS feeds.
2.  Detects and groups semantically similar stories (deduplication/clustering).
3.  Synthesizes a single, comprehensive article from a cluster of related news items using LLMs.
4.  Provides a simple web interface to browse and read these consolidated stories.

## 2. Repository Structure
- **`sumsaar/`**: Contains the main application code (scrapers, Pydantic models, similarity logic).
- **`dags/`**: Contains Apache Airflow DAGs for automation and pipelining.
- **`docker/`**: Contains Docker artifacts, including `docker-compose.yml` and `Dockerfile`s.
- **`docs/`**: Contains documentation, ideas, and conversation history.

## 3. Technical Stack
- **Language**: Python.
- **Orchestration**: Apache Airflow (configured with CeleryExecutor, Redis, and Postgres).
- **Frontend**: Streamlit (running as a service in Docker).
- **Data Ingestion**: `newspaper4k` integrated with Playwright (headless browsing) to handle JavaScript-heavy sites. Also interfaces with a `crawl4ai` Docker container to fetch content as Markdown.
- **LLM Integration**: Uses the `openai` library (replacing direct `ollama` usage) to support both local and paid cloud-based models.
- **Agents**: Combines custom-written AI agents with the `smolagents` library for refining summarization.
- **Data Validation**: Pydantic models used to constrain and validate JSON data structures.

## 4. Key Algorithms & Decisions
- **Similarity Detection**:
    - **Current Approach**: Latent Semantic Analysis (LSA). It has proven faster and more effective for this specific use case than LDA or pure LLM-based pairwise comparisons.
    - **Discarded Approaches**: Latent Dirichlet Allocation (LDA) resulted in too many false groupings.
- **Consolidation**:
    - Uses LLMs to rewrite the set of related articles into a single piece.
    - **Challenges**:
        - Handling "updates" (e.g., evolving casualty numbers in a developing story).
        - Merging dense numerical data without confusion.
        - Processing speed when dealing with large clusters.

## 5. Known Issues & Constraints
- **Scraping**: Playwright integration can be flaky; the code currently handles exceptions by skipping failed downloads.
- **Entity Extraction**: Standard NER tools sometimes miss obvious subject entities (e.g., "gold" in a financial article).
- **Development Philosophy**: Focus on a minimal working prototype first to avoid getting overwhelmed by complex LLM options.

## 6. Development Environment
- The project runs via Docker Compose.
- Services include: Airflow (Webserver, Scheduler, Worker, Triggerer), Postgres, Redis, and Streamlit.
- The `sumsaar` directory is volume-mounted into the containers to facilitate development.