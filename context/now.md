# Now — Immediate Priorities

_Update this at the end of each session with what to tackle next._

## Current State

Railway migration is **complete**. App is live and working with RAG enabled.

- Production URL: `https://clutchai-production.up.railway.app`
- Cron job (`lockedon-cron`) running hourly — populating 655-video LockedOn backlog (~15 videos/run, ~44 hrs total)
- `main` branch is the deploy branch for all Railway services

## Next Steps

1. **GCP Console cleanup** (manual):
   - Cloud SQL `clutchai-db`: disable deletion protection → delete instance
   - Secret Manager: delete OPENAI_API_KEY, YAHOO_CLIENT_ID, YAHOO_CLIENT_SECRET, CLOUDSQL_PASSWORD, YAHOO_ACCESS_TOKEN_JSON
   - Artifact Registry: delete Cloud Run build images
2. **Monitor cron backlog**: Once 655 LockedOn videos are loaded (~44 hrs from now), consider switching cron from hourly (`0 * * * *`) to weekly (`0 3 * * 0`)
3. **Performance optimizations** (from 2026-03-25 notes):
   - Parallelize `UserContextGatherer.gather()` Yahoo API calls (ThreadPoolExecutor, ~60-70% speedup)
   - Parallelize research agents in supervisor
   - Add streaming (`stream()` method exists but unused in `chat()`)

## Blockers

None.
