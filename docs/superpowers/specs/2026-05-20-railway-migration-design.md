# Railway Migration Design

**Date:** 2026-05-20
**Branch:** optimization (or new `railway-migration` branch)
**Scope:** Migrate infrastructure from GCP to Railway. UI (Streamlit) unchanged.

## Goal

Replace GCP (Cloud Run + Cloud SQL + Secret Manager) with Railway. Reduce cost and operational complexity for a solo MVP. The app, agents, and all business logic are untouched — only the deployment infrastructure and database connection layer change.

## Railway Project Structure

Three services in one Railway project:

| Service | Type | Description |
|---------|------|-------------|
| `clutchai-web` | Web service | Streamlit app, deployed from GitHub via Dockerfile |
| `clutchai-db` | PostgreSQL plugin | Railway-managed postgres with pgvector extension |
| `clutchai-cron` | Cron job | Vectorstore update scripts, same Docker image, different start command |

Railway automatically injects `DATABASE_URL` into all services in the same project when the PostgreSQL plugin is attached.

## Code Changes

### 1. `data/postgres/connection.py` — rewrite

Remove the Google Cloud SQL Python Connector entirely. Replace with a standard SQLAlchemy engine built from `DATABASE_URL`.

- Keep the `PostgresConnection` class interface (`get_engine()`, context manager `__enter__`/`__exit__`)
- Read connection from `DATABASE_URL` env var (Railway injects this automatically)
- Use `psycopg2` as the driver (already in requirements or easy to add)
- Remove all GCP-specific params: `project_id`, `region`, `instance`, and the `Connector` class

All consumers (`rag_manager.py`, vectorstore scripts) call `get_engine()` — no changes needed there.

### 2. `Dockerfile` — one-line fix

Railway injects `PORT` dynamically. Change the CMD from hardcoded `8080` to `$PORT`:

```dockerfile
CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port $PORT"]
```

### 3. `requirements.txt` — remove GCP dependencies

`psycopg2-binary` is already present. Remove `cloud-sql-python-connector[pg8000,psycopg2]` — `pg8000` is pulled in transitively and goes away with it.

## Environment Variables

Set these in Railway dashboard (web service + cron service). `DATABASE_URL` is auto-injected by the PostgreSQL plugin.

| Variable | Value |
|----------|-------|
| `OPENAI_API_KEY` | from current Secret Manager |
| `YAHOO_CLIENT_ID` | from current Secret Manager |
| `YAHOO_CLIENT_SECRET` | from current Secret Manager |
| `YAHOO_ACCESS_TOKEN_JSON` | from current Secret Manager |
| `YAHOO_LEAGUE_ID` | `58930` |
| `YAHOO_REDIRECT_URI` | Railway app URL (or custom domain) |
| `RUNTIME_ENVIRONMENT` | `docker` |
| `DISABLE_RAG` | `false` |

GCP-specific vars (`GOOGLE_CLOUD_PROJECT`, `CLOUDSQL_*`) are removed.

## Database Setup

1. Add Railway PostgreSQL plugin to the project
2. Enable `pgvector` extension: `CREATE EXTENSION IF NOT EXISTS vector;`
3. Re-run vectorstore pipeline scripts against the new `DATABASE_URL` to populate the knowledge base

The schema and pipeline scripts are unchanged — they just point to a different database.

## Cron Job

The cron service uses the same Docker image as the web service with a different start command. Three update scripts exist:

- `scripts/pipelines/update_base_knowledge.py` — base knowledge base (articles, static content)
- `scripts/pipelines/update_lockedon_knowledge.py` — LockedOn podcast transcripts
- `scripts/pipelines/update_vector_database.py` — full vectorstore rebuild

Run these as separate Railway cron jobs or wrap them in a single shell script. Schedule: weekly or as needed. `DATABASE_URL` is shared automatically within the Railway project.

## What Gets Removed

- `scripts/pipelines/deploy.sh` — Railway deploys on push to GitHub
- `scripts/pipelines/update_secrets.sh` — replaced by Railway dashboard env vars
- `scripts/pipelines/grant_secrets_access.sh` — not needed
- GCP dependencies: `cloud-sql-python-connector`, `pg8000`, `google-cloud-sql-connector`

Other `scripts/pipelines/` scripts (vectorstore pipeline, knowledge base updates) are kept — they run as-is in the cron job context.

## What Stays the Same

- All agent code (`agents/`)
- All tools (`agents/tools/`)
- RAG manager (`agents/rag/`)
- Streamlit app (`app/streamlit_app.py`)
- All config YAML files
- Docker scripts (`scripts/docker/`)
- Dockerfile structure (one-line CMD change only)
