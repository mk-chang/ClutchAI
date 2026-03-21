# GCP Configuration & Deployment Notes

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

Connection uses `cloud-sql-python-connector` (pg8000/psycopg2). See `data/cloud_sql/connection.py`.

## Deployment

Deploy from project root:
```bash
./scripts/gcloud/deploy.sh                  # deploy only
./scripts/gcloud/deploy.sh --update-secrets # sync .env to Secret Manager first
```

### Secrets in Secret Manager
Required secrets (must exist before deploy):
- `OPENAI_API_KEY`
- `YAHOO_CLIENT_ID`
- `YAHOO_CLIENT_SECRET`
- `CLOUDSQL_PASSWORD`
- `YAHOO_ACCESS_TOKEN_JSON` — required for Yahoo OAuth on Cloud Run (no interactive login). Generate locally first via normal OAuth flow, then run `scripts/gcloud/build_yahoo_token_json.py`.

Optional secrets (commented out in deploy.sh):
- `GOOGLE_CLOUD_KEY`, `HASHTAG_BASKETBALL_USERNAME`, `HASHTAG_BASKETBALL_PASSWORD`, `LANGSMITH_API_KEY`, `FIRECRAWL_API_KEY`

### Non-sensitive env vars (set directly in deploy.sh)
`YAHOO_REDIRECT_URI`, `YAHOO_LEAGUE_ID`, `RUNTIME_ENVIRONMENT=docker`, `GOOGLE_CLOUD_PROJECT`, `CLOUDSQL_*`, `CLOUDSQL_USER`

## IAM / Access
Run `./scripts/gcloud/grant_secrets_access.sh` to grant Cloud Run service account access to secrets. Only needed once or when adding new secrets.

## Notes
- `RUNTIME_ENVIRONMENT=docker` tells yfpy not to open a browser for OAuth (uses token JSON instead)
- `DISABLE_RAG=true` in `.env` skips Cloud SQL for local dev without GCP credentials
