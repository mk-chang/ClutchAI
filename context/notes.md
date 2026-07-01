# Notes — Decisions & Gotchas

_Running log of non-obvious decisions, debugging findings, and things to remember. Prepend new entries._

---

## 2026-07-01

- LangSmith project was renamed `pr-roasted-deliberation-80` → `clutch-ai` (hyphen). Search `list_projects` with `project_name="clutch"`, not `"clutch_ai"` — underscore won't match.
- `agents/tools/waiver_wire.py` (WaiverWireTool / `get_waiver_wire_players`) exists on `staging` and `feature/waiver_wire` but is **absent from `feature/player_db` and `main`** — deployed/staging behavior can diverge from what's in this branch's `agents/tools/yahoo_api.py`.
- `BaseAgent` (base_agent.py) already supports `user_context` end-to-end (constructor param + `_enhance_system_prompt`) for every agent — but `multi_agent_system.py` only wires it into `SupervisorAgent`/`FantasyAnalystAgent`, not `YahooFantasyAgent`/`StatisticAgent`. Giving those two agents the already-known league_key/roster context needs zero new plumbing, just one line at construction.

---

## 2026-06-30

- **Deployment flow:** merge feature branch → staging first, manually verify, then merge to main for production. Never merge a feature branch directly to main.
- `RAILWAY_TOKEN` does NOT work for Railway MCP auth — never suggest it or add it to settings.json. When MCP returns Unauthorized, tell user to run `railway login` in terminal.

---

## 2026-05-25

- Railway cron `&&` in start command is not interpreted — Railway doesn't wrap in a shell; use `bash -c "cmd1 && cmd2"` or a single Python script
- GCP Cloud SQL suspended instance can't be patched (HTTP 409) — delete deletion protection via GCP Console, not CLI
- `update_vector_database.py` path was broken: needed `.parent.parent.parent` (3 levels up from `scripts/pipelines/`), and import was `scripts.vectordb_pipelines` (stale) → `scripts.pipelines`
- Railway MCP session expires frequently — fix: `railway login` in terminal + VS Code Reload Window

---

## 2026-05-20

- Migrating to Railway: `CLOUDSQL_VECTOR_TABLE` → `VECTOR_TABLE`, `CLOUDSQL_APP_TABLE` → `APP_TABLE` in env vars
- Railway PostgreSQL injects `DATABASE_URL` automatically — don't set it manually in Railway dashboard
- `scripts/pipelines/create_database.py` deleted (Railway manages DB creation)
- `YAHOO_ACCESS_TOKEN_JSON` migration: generate locally via `scripts/pipelines/build_yahoo_token_json.py`, paste JSON blob into Railway env var

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
- Yahoo OAuth on Cloud Run requires `YAHOO_ACCESS_TOKEN_JSON` secret — generate locally first via `scripts/pipelines/build_yahoo_token_json.py`
- Cloud SQL password is optional if using IAM authentication
- `update_secrets.sh` had a wrong `PROJECT_ROOT` path (one level too shallow) — fixed
- All agents use `temperature=0` and `max_tokens=150000` to stay within 200k TPM limit
