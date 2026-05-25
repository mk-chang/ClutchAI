# Session: Railway Migration Complete
_2026-05-25_

## What Was Accomplished

### 1. Static Knowledge Pipeline (update_base_knowledge.py)
- Ran `update_base_knowledge.py` pointed at Railway Postgres using `DATABASE_PUBLIC_URL`
- Confirmed it was already run manually — all entries skipped (already in vectorstore)
- Added `update_base_knowledge.py` to the cron job sequence

### 2. GCP Teardown (partial)
- Deleted Cloud Run service `clutchai` (us-central1) — confirmed deleted
- Cloud SQL instance `clutchai-db` is SUSPENDED and has deletion protection enabled
  - `gcloud sql instances patch ... --no-deletion-protection` failed with HTTP 409 (instance suspended, can't patch)
  - Must be done via GCP Console manually: disable deletion protection → delete
- Secret Manager secrets: Secret Manager API blocked due to billing disabled on project — delete via GCP Console
- Artifact Registry: delete via GCP Console

### 3. Enabled RAG on Railway
- Flipped `DISABLE_RAG=false` on ClutchAI service via Railway MCP
- Triggered redeploy — app verified working end-to-end by user

### 4. Merged railway_migration → main
- `railway_migration` branch was 9 commits ahead of main
- Remote main had a diverged commit (from an optimization branch PR merge)
- Pulled with rebase, merged fast-forward, pushed successfully
- All Railway services now deploy from `main`

### 5. Cron Job Fixes
- **Problem 1**: `&&` in start command wasn't being interpreted — Railway doesn't wrap in a shell
  - Fix: Changed to `bash -c "cmd1 && cmd2"`
- **Problem 2**: `update_base_knowledge.py` was skipping all YouTube videos — those are static seed entries already loaded; LockedOn episodes come from `update_lockedon_knowledge.py`
- **Problem 3**: User requested consolidating to single `update_vector_database.py` script
  - Fixed broken `project_root` path in that script (was `.parent.parent`, needed `.parent.parent.parent`)
  - Fixed broken import path (was `scripts.vectordb_pipelines`, now `scripts.pipelines`)
  - Updated Railway cron start command to `python scripts/pipelines/update_vector_database.py`

### 6. Directory Renames
- `data/cloud_sql/` → `data/postgres/`
- `scripts/gcloud/` → `scripts/pipelines/`
- Updated all references across 29 files (Python imports, markdown, shell scripts, CLAUDE.md)
- Committed as `6c58f53`

## Key Decisions

- **Cron runs `update_vector_database.py`** (master script) rather than chaining individual scripts — cleaner, handles errors independently per pipeline, logs duration breakdown
- **Merged to main** rather than changing Railway branch per-service — simpler long-term, migration is complete
- **GCP Console teardown** for Cloud SQL + secrets (CLI blocked by suspended billing/instance state)

## Problems & Fixes

| Problem | Fix |
|---------|-----|
| Railway cron `&&` not interpreted | Wrapped in `bash -c "..."`, then switched to single Python script |
| `update_vector_database.py` broken path | `.parent.parent` → `.parent.parent.parent` |
| `update_vector_database.py` wrong import (`vectordb_pipelines`) | Updated to `scripts.pipelines` |
| Railway MCP session expiring repeatedly | `railway login` + VS Code Reload Window (Cmd+Shift+P) |
| Cloud SQL deletion protection | Must use GCP Console (CLI can't patch suspended instance) |

## Files Changed

- `data/postgres/` — renamed from `data/cloud_sql/`
- `scripts/pipelines/` — renamed from `scripts/gcloud/`
- `scripts/pipelines/update_vector_database.py` — fixed project_root and import path
- All import references updated across agents, tests, context docs

## Commits This Session

- `6c58f53` — Rename data/cloud_sql -> data/postgres, scripts/gcloud -> scripts/pipelines
