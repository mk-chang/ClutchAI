# Now — Immediate Priorities

_Update this at the end of each session with what to tackle next._

## Current State

Railway migration complete. Transcript cleaning feature implemented and deployed to staging. Waiver wire tool implemented with Postgres-persisted cache (on `feature/waiver_wire` branch, 7 commits ahead of origin). Player stats database design in progress (brainstorming ~70% complete).

- Production URL: `https://clutchai-production.up.railway.app`
- Staging URL: `https://clutchai-staging.up.railway.app`
- `main` → production auto-deploy | `staging` → staging auto-deploy (once branch source is set)
- Cron job (`lockedon-cron`) still running hourly in production ingesting LockedOn backlog

## Next Steps

1. **Resume player stats DB brainstorm** — design is ~70% done (schema + pipeline approved). Need to:
   - Present Section 3: Agent tools (pre-built tools + raw SQL)
   - Write spec to `docs/superpowers/specs/2026-06-29-player-stats-db-design.md`
   - Run spec self-review → user review → invoke `writing-plans`
   - See session notes: `context/sessions/2026-06-29-player-stats-db-brainstorm.md`
2. **Push `feature/waiver_wire`** — 7 commits ahead of origin
3. **Manual: Set staging branch source in Railway dashboard**
   - clutchai project → staging env → ClutchAI service → Settings → Source → change `main` → `staging`
   - Do same for `lockedon-cron` service
4. **Set `DEV_MODE=true` on staging `lockedon-cron`** via Railway dashboard
5. **Monitor cron backlog**: Once 655 LockedOn videos loaded, switch production cron from hourly to weekly (`0 3 * * 0`)
6. **GCP Console cleanup** (manual, low urgency):
   - Cloud SQL `clutchai-db`: disable deletion protection → delete
   - Secret Manager: delete 5 secrets
   - Artifact Registry: delete Cloud Run build images
7. **Performance optimizations** (from 2026-03-25 notes):
   - Parallelize `UserContextGatherer.gather()` Yahoo API calls (ThreadPoolExecutor, ~60-70% speedup)
   - Parallelize research agents in supervisor
   - Add streaming (`stream()` method exists but unused in `chat()`)

## Blockers

None.
