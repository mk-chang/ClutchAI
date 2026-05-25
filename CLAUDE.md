# ClutchAI

Multi-agent fantasy basketball assistant. Supervisor routes queries to specialized research agents, then synthesizes results via an analysis agent.

## Where Things Live

| What | Path | Description |
|------|------|-------------|
| Agent implementations | `agents/multi_agent/` | Supervisor, Yahoo, Stats, News, and Analysis agents |
| Tools | `agents/tools/` | Yahoo API (45 tools), NBA API (16 tools), RSS, news, rankings |
| RAG | `agents/rag/` | pgvector retrieval from Cloud SQL knowledge base |
| UI | `app/streamlit_app.py` | Streamlit frontend, entry point |
| Database | `data/postgres/` | Cloud SQL connection and table schema |
| Config | `config/` | YAML config for agents, RAG, vectorstore, tools, UI |
| Tests | `tests/` | Integration tests (pytest) |
| Docs | `docs/` | Deployment guide, Yahoo OAuth setup |
| Deploy scripts | `scripts/pipelines/` | Cloud Run deploy, vectorstore pipeline, secrets sync |
| Docker scripts | `scripts/docker/` | Local build and run |
| Context | `context/` | Session notes, priorities, agent patterns, GCP details |