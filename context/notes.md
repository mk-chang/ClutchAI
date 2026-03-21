# Notes — Decisions & Gotchas

_Running log of non-obvious decisions, debugging findings, and things to remember. Prepend new entries._

---

## 2026-03-21

- `DISABLE_RAG=true` in `.env` lets you run locally without GCP/Cloud SQL — Yahoo, stats, news, and RSS still work
- `RUNTIME_ENVIRONMENT=docker` must be set in Cloud Run to prevent yfpy from opening a browser for OAuth
- Yahoo OAuth on Cloud Run requires `YAHOO_ACCESS_TOKEN_JSON` secret — generate locally first via `scripts/gcloud/build_yahoo_token_json.py`
- Cloud SQL password is optional if using IAM authentication
- `update_secrets.sh` had a wrong `PROJECT_ROOT` path (one level too shallow) — fixed
- All agents use `temperature=0` and `max_tokens=150000` to stay within 200k TPM limit
