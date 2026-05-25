# Session — 2026-05-20 — Railway Migration

## What We Did

### Brainstorming & Design
- Decided to migrate from GCP (Cloud Run + Cloud SQL + Secret Manager) to Railway
- Motivation: solo MVP, GCP too expensive and complex for the use case
- Key decisions made:
  - **UI**: Keep Streamlit (no reason to swap for Railway migration)
  - **Database**: Railway PostgreSQL plugin (pgvector supported, DATABASE_URL auto-injected, cron job access built-in)
  - **Dropped Gradio** — briefly considered but keeping Streamlit was the right call
- Wrote design spec: `docs/superpowers/specs/2026-05-20-railway-migration-design.md`
- Wrote implementation plan: `docs/superpowers/plans/2026-05-20-railway-migration.md`

### Code Implementation (Tasks 1–4, subagent-driven)

**Task 1: Rewrote `data/postgres/connection.py`**
- Replaced 220-line GCP-specific Cloud SQL Connector implementation with 33-line standard psycopg2 class
- New class reads `DATABASE_URL` env var, rewrites `postgresql://` → `postgresql+psycopg2://` for SQLAlchemy
- Deleted `create_database_if_not_exists` function (Railway manages DB creation)
- Deleted `scripts/pipelines/create_database.py` (imported the deleted function, now obsolete)
- Created `tests/test_postgres_connection.py` with 5 unit tests (no real DB needed)

**Task 2: Updated `data/postgres/schema.py` + `tests/test_vectordb_connection.py`**
- Renamed `CLOUDSQL_VECTOR_TABLE` → `VECTOR_TABLE` in `get_default_table_name()`
- Renamed `CLOUDSQL_APP_TABLE` → `APP_TABLE` in `get_app_table_name()`
- Updated all docstring references to the old env var names
- Updated `test_vectordb_connection.py` skipif: now checks `DATABASE_URL` instead of 5 GCP vars

**Task 3: Updated `requirements.txt`**
- Removed `cloud-sql-python-connector[pg8000,psycopg2]`
- Updated comment from "Google Cloud SQL" to just "PostgreSQL with pgvector"
- `psycopg2-binary` and `langchain-postgres` remain

**Task 4: Updated `Dockerfile`**
- Changed CMD from hardcoded port 8080 to `${PORT:-8080}` — Railway injects `PORT` dynamically
- Fallback to 8080 keeps local Docker runs working

## Files Changed

| File | Change |
|------|--------|
| `data/postgres/connection.py` | Full rewrite — Cloud SQL Connector → psycopg2 + DATABASE_URL |
| `tests/test_postgres_connection.py` | Created — 5 unit tests for new connection class |
| `data/postgres/schema.py` | Env var renames: CLOUDSQL_VECTOR_TABLE→VECTOR_TABLE, CLOUDSQL_APP_TABLE→APP_TABLE |
| `tests/test_vectordb_connection.py` | Updated skipif to check DATABASE_URL |
| `requirements.txt` | Removed cloud-sql-python-connector |
| `Dockerfile` | CMD uses ${PORT:-8080} |
| `scripts/pipelines/create_database.py` | Deleted (imported removed function, obsolete on Railway) |
| `docs/superpowers/specs/2026-05-20-railway-migration-design.md` | Created |
| `docs/superpowers/plans/2026-05-20-railway-migration.md` | Created |

## State Left In

- On `railway_migration` branch, 6 commits ahead of main
- `docs/superpowers/` is untracked (uncommitted) — commit and push needed
- All code changes complete and tested
- Tasks 5–8 remain: Railway dashboard setup (manual), vectorstore population, cron job, verification

## Notes

- The `CLOUDSQL_VECTOR_TABLE` → `VECTOR_TABLE` rename means Railway env vars must use the new names
- Yahoo OAuth: `YAHOO_ACCESS_TOKEN_JSON` still works the same — generate locally, paste into Railway env vars
- The 3 vectorstore update scripts in `scripts/pipelines/` are kept and will run as Railway cron jobs
- User installed Railway CLI and Railway MCP into the clutchai conda env this session — restart required for MCP to be available
