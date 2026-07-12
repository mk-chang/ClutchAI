# Now — Immediate Priorities

_Update this at the end of each session with what to tackle next._

## Current State

Railway migration is **complete**, app live. A `/superpowers:brainstorming` session for 3 multi-agent workflow gaps (found via LangSmith review) is **paused mid-investigation, no design doc written yet**.

- Production URL: `https://clutchai-production.up.railway.app`
- `main` branch is the deploy branch for all Railway services
- Currently on `feature/player_db` branch

## Next Steps

1. **Resume the paused workflow-gap brainstorm** (see `context/sessions/2026-07-01-langsmith-review-and-workflow-brainstorm.md` for full detail):
   - Run `git show staging:agents/tools/waiver_wire.py` to check whether the waiver/free-agent tool can accept a pre-known `league_key` or has the same self.query-priming constraint as `yahoo_api.py`'s league-level tools — this was the open question when paused.
   - All 4 scoping/approach questions are already answered (bundle all 3 gaps into one spec; Gap 2 = prompt clarification only; Gap 3 = supervisor-side stats call before analysis; Gap 1 = pass `user_context` into `YahooFantasyAgent`, reusing existing `BaseAgent` plumbing).
   - Once waiver_wire.py is checked: recap approaches → present design sections → write spec to `docs/superpowers/specs/2026-07-01-multi-agent-workflow-fixes-design.md` → get approval → `writing-plans`.
2. **GCP Console cleanup** (manual, unverified for a while — may already be done):
   - Cloud SQL `clutchai-db`: disable deletion protection → delete instance
   - Secret Manager: delete OPENAI_API_KEY, YAHOO_CLIENT_ID, YAHOO_CLIENT_SECRET, CLOUDSQL_PASSWORD, YAHOO_ACCESS_TOKEN_JSON
   - Artifact Registry: delete Cloud Run build images
3. **Verify LockedOn cron backlog status** — estimated complete ~2026-05-27 (655 videos @ ~15/run hourly), not reconfirmed since. If done, switch cron from hourly (`0 * * * *`) to weekly (`0 3 * * 0`).
4. **Performance optimizations** (from 2026-03-25 notes, still unaddressed):
   - Parallelize `UserContextGatherer.gather()` Yahoo API calls (ThreadPoolExecutor, ~60-70% speedup)
   - Parallelize research agents in supervisor
   - Add streaming (`stream()` method exists but unused in `chat()`)

## Blockers

None active — brainstorming was paused by user request, not blocked on anything external.
