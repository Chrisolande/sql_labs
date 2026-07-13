# CareerPilot AI Agent

An agentic job discovery, matching, and application workflow system built on **LangGraph**, **FastAPI**, **SQLAlchemy** (PostgreSQL + pgvector), and the **Google Gemini API**.

> [!IMPORTANT]
> This repository houses a production-grade AI agent that processes raw job postings, parses and normalizes details, generates vector embeddings, stores semantic memories of user profile histories, and executes a multi-step LangGraph verification state machine to match candidates with their dream jobs.

---

## Key Features

- **Ingestion Daemon**: Periodic crawler that discovery-seeds job postings from the Job Data Lake, normalizing details and saving them to the database.
- **Jina Reader Processing**: Fetches full-text job listings concurrently using Jina's reader service, utilizing local rate-limit control and dynamic database persistence.
- **Semantic Vector Search**: Denormalizes and embeds job details using `gemini-embedding-2` for Pgvector hybrid retrieval (lexical BM25 + semantic vector).
- **Career Memory Sync**: Copies relational user profile details (education, skills, work history) directly into LangGraph's persistent Postgres Store for personalized LLM context.
- **LangGraph State Machine**: A resilient QA routing workflow featuring query analysis, hybrid retrieval, candidate scoring, draft generation, and human-in-the-loop application submission.

---

## Tech Stack

- **Core Framework**: LangGraph, LangChain
- **Language Models**: Google Gemini (`gemini-2.5-flash` for fast scoring, `gemini-2.5-pro` for reasoning)
- **Database**: PostgreSQL with `pgvector` extension
- **Database ORM**: SQLAlchemy 2.0 with async pg drivers and Alembic migrations
- **Dependency & Package Manager**: `uv`

---

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.12+ (or managed via `uv`)

### Environment Setup

Create a `.env` file in the project root:

```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/careerpilot
GOOGLE_GENERATIVE_AI_API_KEY=your_gemini_api_key_here
JINA_API_KEY=your_jina_api_key_here
JOB_DATA_LAKE_API_KEY=your_job_data_lake_key_here
```

### Installation & Migrations

1. **Spin up the database**:
   ```bash
   docker compose up -d db
   ```

2. **Apply migrations**:
   ```bash
   uv run alembic upgrade head
   ```

---

## Running the Ingestion Pipeline

To run the job ingestion daemon to sync job listings and fill up embeddings:

```bash
# Run the daemon loop to discover jobs
uv run python app/scripts/jdl_daemon.py
```

> [!TIP]
> The Jina Reader description fetcher and Gemini Embeddings populator are both fully rate-limit protected. If they exceed API quotas, they gracefully back off, sleep, and commit successful progress to the database incrementally.

---

## Running the LangGraph Agent

You can instantiate and invoke the QA graph state machine programmatically:

```python
import asyncio
from app.graph.agent import build_qa_graph_postgres

async def run():
    async with build_qa_graph_postgres() as graph:
        config = {"configurable": {"thread_id": "unique_session_thread"}}
        state = {
            "user_query": "Looking for a hybrid Python Backend Engineer job",
            "user_id": "00000000-0000-0000-0000-000000000123"
        }
        result = await graph.ainvoke(state, config)
        print("Draft response:", result.get("draft_response").summary)

asyncio.run(run())
```

---

## Architecture & Database Schema

The relational database is split into:
- **`jobs` & `job_sources`**: Raw and normalized job metadata.
- **`job_descriptions`**: Crawled and cleaned job details.
- **`embeddings`**: 1024-dimensional semantic document vectors.
- **`career_memory`**: User skills, projects, achievements, and employment history.
- **`applications`**: Active job application drafts, critiques, and submission states.
- **`agent_tasks`**: Task status queues for background graph execution.
