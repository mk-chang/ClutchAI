# Notes — Decisions & Gotchas

_Running log of non-obvious decisions, debugging findings, and things to remember. Prepend new entries._

---

## 2026-05-20

- Migrating to Railway: `CLOUDSQL_VECTOR_TABLE` → `VECTOR_TABLE`, `CLOUDSQL_APP_TABLE` → `APP_TABLE` in env vars
- Railway PostgreSQL injects `DATABASE_URL` automatically — don't set it manually in Railway dashboard
- `scripts/gcloud/create_database.py` deleted (Railway manages DB creation)
- `YAHOO_ACCESS_TOKEN_JSON` migration: generate locally via `scripts/gcloud/build_yahoo_token_json.py`, paste JSON blob into Railway env var

---

## 2026-03-25

- Agent init is slow because `UserContextGatherer.gather()` makes sequential Yahoo API calls on every new session — parallelizing with `ThreadPoolExecutor` should cut ~60-70%
- Cloud Run likely scales to zero — `min-instances=1` in deploy config will eliminate cold starts
- `stream()` method exists on all agents but `chat()` in `multi_agent_system.py` doesn't use it — easy streaming win
- Research agents in supervisor are called sequentially — can be parallelized

---

## 2026-03-21

- `DISABLE_RAG=true` in `.env` lets you run locally without GCP/Cloud SQL — Yahoo, stats, news, and RSS still work
- `RUNTIME_ENVIRONMENT=docker` must be set in Cloud Run to prevent yfpy from opening a browser for OAuth
- Yahoo OAuth on Cloud Run requires `YAHOO_ACCESS_TOKEN_JSON` secret — generate locally first via `scripts/gcloud/build_yahoo_token_json.py`
- Cloud SQL password is optional if using IAM authentication
- `update_secrets.sh` had a wrong `PROJECT_ROOT` path (one level too shallow) — fixed
- All agents use `temperature=0` and `max_tokens=150000` to stay within 200k TPM limit
