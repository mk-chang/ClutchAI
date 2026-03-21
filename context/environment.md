# Local Environment

## Conda Environment

| Key | Value |
|-----|-------|
| Env name | `ClutchAI` |
| Python | 3.11 |
| Setup | `conda env create -f environment.yml` |
| Activate | `conda activate ClutchAI` |

## Key Dependencies

- `langgraph` — multi-agent orchestration
- `langchain` / `langchain-openai` — LLM + RAG
- `langchain-postgres` — pgvector integration
- `cloud-sql-python-connector[pg8000,psycopg2]` — Cloud SQL connection
- `yfpy` — Yahoo Fantasy API
- `nba_api` — NBA stats
- `streamlit` — frontend

## Environment Variables

Copy `env.example` → `.env` and fill in credentials. Key variables:
- `DISABLE_RAG=true` — run without Cloud SQL (local dev without GCP)
- `RUNTIME_ENVIRONMENT=docker` — disables browser OAuth (set automatically in Cloud Run)
- `OPENAI_MODEL` — defaults to `gpt-4` in env.example, but agents use `gpt-4o` / `gpt-4o-mini` from `config/multiagent_config.yaml`

## Notes

_Update this section with any local setup quirks discovered over time._
