# Now — Immediate Priorities

_Update this at the end of each session with what to tackle next._

## Next Steps

Railway migration in progress — code changes done (Tasks 1–4), infra setup remains.

Plan: `docs/superpowers/plans/2026-05-20-railway-migration.md`

1. **Task 5** — Railway project setup (manual):
   - Create Railway project + PostgreSQL plugin
   - Enable pgvector extension
   - Set env vars (see plan for full list — note `VECTOR_TABLE` not `CLOUDSQL_VECTOR_TABLE`)
   - Connect GitHub repo and deploy
2. **Task 6** — Populate vectorstore: run `update_base_knowledge.py`, `update_lockedon_knowledge.py`, `update_vector_database.py` against Railway postgres
3. **Task 7** — Set up Railway cron job for weekly vectorstore updates
4. **Task 8** — End-to-end verification + GCP teardown

## Blockers

- Restart Claude Code session first to activate Railway MCP (installed this session)
