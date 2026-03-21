# ClutchAI — Claude Code Guide

## Project Overview

Multi-agent fantasy basketball assistant. Users ask natural language questions about their Yahoo Fantasy league and get data-grounded answers. The system routes queries through a **Supervisor → specialized Research agents → Analysis agent** pipeline, pulling from the Yahoo Fantasy API, NBA API, a pgvector knowledge base (Google Cloud SQL), and RSS news feeds. The Streamlit frontend is deployed on Google Cloud Run.

See `README.md` for full architecture diagrams.

---

## Architecture

```
Supervisor Agent (gpt-4o)
├── Yahoo Fantasy Agent (gpt-4o-mini)  — league/team/player data via Yahoo API
├── Statistic Agent (gpt-4o-mini)      — NBA stats via nba_api
├── News Agent (gpt-4o-mini)           — RSS feeds, dynasty rankings, knowledge base
└── Analysis Agent (gpt-4o)            — synthesizes research into recommendations
```

Key files:
- `config/multiagent_config.yaml` — system prompts and model config for all agents
- `agents/multi_agent/` — agent implementations
- `agents/tools/` — all tool implementations (Yahoo, NBA, RAG, news, etc.)
- `app/streamlit_app.py` — entry point

---

## Key Commands

```bash
# Run the app locally
streamlit run app/streamlit_app.py

# Run tests
pytest

# Run specific test file
pytest tests/<file>.py -k '<function_name>'

# Docker local build and run
./scripts/docker/local_build.sh
./scripts/docker/local_run.sh
./scripts/docker/rebuild_and_run.sh

# Deploy to Google Cloud Run
./scripts/gcloud/deploy.sh

# Deploy and sync .env secrets first
./scripts/gcloud/deploy.sh --update-secrets

# Update vectorstore from YouTube (LockedOn podcasts)
python scripts/gcloud/update_lockedon_knowledge.py

# Update vectorstore from articles
python scripts/gcloud/update_vector_database.py

# Sync .env to GCP Secret Manager
./scripts/gcloud/update_secrets.sh
```

---

## Configuration Files

| File | Controls |
|------|----------|
| `config/multiagent_config.yaml` | Agent system prompts, model names, token limits |
| `config/agent_config.yaml` | Single-agent config |
| `config/rag_config.yaml` | RAG chunk size, retrieval k, search type |
| `config/vector_config.yaml` | Vectorstore pipeline (max videos, rate limiting) |
| `config/tools_config.yaml` | Tool-level configuration |
| `config/streamlit_config.yaml` | UI configuration |

---

## External Services

| Service | Purpose | Details |
|---------|---------|---------|
| Google Cloud Run | App hosting | Project: `clutchai-480619`, Region: `us-central1` |
| Google Cloud SQL | pgvector knowledge base | Instance: `clutchai-db`, DB: `clutchai_db` |
| GCP Secret Manager | Secrets in production | Synced via `update_secrets.sh` |
| Yahoo Fantasy API | League/team/player data | OAuth via yfpy library |
| OpenAI | LLM + embeddings | GPT-4o (supervisor/analysis), GPT-4o-mini (research) |
| LangSmith | LLM tracing | Optional, toggle via `LANGCHAIN_TRACING_V2` |

See `context/cloud/gcp.md` for deployment details and known quirks.
See `docs/DEPLOYMENT.md` for the full deployment guide.
See `docs/YAHOO_API_SETUP_GUIDE.md` for Yahoo OAuth setup.

**Local dev without GCP:** Set `DISABLE_RAG=true` in `.env` to skip Cloud SQL. Yahoo, stats, news, and RSS features still work.

---

## Development Guide

### Adding a New Tool
1. Create the tool in `agents/tools/<category>.py` following the pattern in `agents/tools/base.py`
2. Register it in the appropriate agent in `agents/multi_agent/`
3. Add integration test in `tests/`

### Adding a New Agent
1. Subclass `BaseAgent` in `agents/multi_agent/base_agent.py`
2. Add system prompt and model config to `config/multiagent_config.yaml`
3. Wire into `agents/multi_agent/supervisor.py`

### Environment Setup
```bash
# Create conda environment
conda env create -f environment.yml
conda activate ClutchAI

# Or install via pip
pip install -r requirements.txt
```

Python version: 3.11

### Running Without GCP
Set `DISABLE_RAG=true` in `.env` to run locally without Cloud SQL credentials.

---

## Context Folder

`context/` is a local-only folder (gitignored) where Claude maintains working knowledge about this project. It is not pushed to git.

- `context/cloud/gcp.md` — GCP project details, deployment notes, known issues
- `context/agents/patterns.md` — agent quirks, prompt patterns, lessons learned
- `context/data/vectorstore.md` — vectorstore state, pipeline run notes
- `context/environment.md` — local env setup, conda env, known quirks
