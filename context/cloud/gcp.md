# GCP Configuration & Deployment Notes

> **Status:** Being replaced by Railway. GCP still active — do not tear down until Railway is verified (Task 8 of migration plan).

## Project

| Key | Value |
|-----|-------|
| Project ID | `clutchai-480619` |
| Region | `us-central1` |
| Cloud Run service | `clutchai` |
| Production URL | `https://www.clutchai.app` |

## Cloud SQL

| Key | Value |
|-----|-------|
| Instance | `clutchai-db` |
| Database | `clutchai_db` |
| Vector table | `vectorstore` |
| App table | `app` |
| User | `clutchai_user` |
| Auth | Password or IAM (password optional if using IAM) |

~~Connection uses `cloud-sql-python-connector` (pg8000/psycopg2). See `data/postgres/connection.py`.~~
Connection now uses standard psycopg2 via `DATABASE_URL`. The Cloud SQL Connector has been removed.

## Deployment (legacy — use Railway going forward)

Deploy from project root:
```bash
./scripts/pipelines/deploy.sh                  # deploy only
./scripts/pipelines/deploy.sh --update-secrets # sync .env to Secret Manager first
```

### Secrets in Secret Manager
Required secrets (must exist before deploy):
- `OPENAI_API_KEY`
- `YAHOO_CLIENT_ID`
- `YAHOO_CLIENT_SECRET`
- `CLOUDSQL_PASSWORD`
- `YAHOO_ACCESS_TOKEN_JSON` — required for Yahoo OAuth on Cloud Run (no interactive login). Generate locally first via normal OAuth flow, then run `scripts/pipelines/build_yahoo_token_json.py`.

## Notes
- `RUNTIME_ENVIRONMENT=docker` tells yfpy not to open a browser for OAuth (uses token JSON instead)
- `DISABLE_RAG=true` in `.env` skips Cloud SQL for local dev without GCP credentials
- GCP teardown: `gcloud run services delete clutchai --region us-central1` — only after Railway is verified
