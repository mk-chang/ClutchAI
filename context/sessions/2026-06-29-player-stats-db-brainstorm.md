# Session: Player Stats Database Design — Brainstorming

**Date:** 2026-06-29
**Branch:** feature/waiver_wire
**Type:** Brainstorming (no code written)

## What Was Done

Ran `/superpowers:brainstorming` to design a player stats database layer for ClutchAI. The goal: replace live NBA API calls with a persistent Postgres database to enable data wrangling, trend analysis, and SQL tools for the agent.

The brainstorming session completed through Schema (Section 1) and Pipeline (Section 2) of the design presentation. Agent tools (Section 3) was not reached — session was saved before that section.

## Context Explored

- `data/postgres/player_stats.py` — `PlayerStatsManager` already exists with 3 season-aggregate tables (`player_stats_pg`, `player_stats_total`, `player_stats_p36`), each keyed by `(player_id, season)`.
- `docs/superpowers/plans/2026-05-26-player-stats-database.md` — prior plan for the aggregate stats layer (daily cron + `basketball_monster_stats` tool).
- `agents/tools/nba_api.py` — 16 NBA API tools; existing live-pull approach.
- `data/redis/` — Redis connection already built for waiver wire caching.

## Key Decisions Made

### 1. Database-first approach confirmed
Pull from NBA API once daily; cache in Postgres. Agents query the DB, not the API. Rationale: NBA API rate-limits aggressively (~1s+ per call), repeated live queries make the agent slow and brittle.

### 2. Table scope
Four tables total:

| Table | Scope | Update |
|-------|-------|--------|
| `player_stats_pg/total/p36` (existing) | All seasons | Daily in-season |
| `player_game_logs` | Current season only | Daily in-season |
| `team_schedules` | Current season | Once at season start + postponement patches |
| `opponent_defense_rankings` | Current season | Weekly (Monday) |

**Explicitly excluded:** historical game logs (season aggregates cover dynasty needs), player injury/status (live from Yahoo), league roster/ownership (live from Yahoo).

### 3. Game logs = current season only
Dynasty/keeper analysis is well-served by season aggregates. Game logs serve in-season decisions: waiver wire, start/sit, recent trends. At season end: archive/truncate game logs; std dev persists in aggregate tables.

### 4. Std dev added to season aggregate tables
Compute std dev per stat from game logs during daily update; write back to `player_stats_pg/total/p36`. At season end, std dev is preserved in aggregates even after game logs are cleaned up. Enables consistency analysis without permanent game log storage.

### 5. Opponent defense rankings table
Weekly `LeagueDashTeamStats` (opponent mode) snapshot. Enables "favorable matchup" analysis: not just games remaining but quality of those matchups. Key for streaming pickup recommendations.

### 6. Agent SQL interface: Hybrid (Option C)
- Pre-built tools for common waiver wire queries: `get_recent_form`, `get_schedule_density`, `get_season_trends`
- Raw SQL fallback (`query_stats_db(sql)`) for novel/complex queries
- Rationale: fast path for 80% of cases, flexible for the tail

### 7. Pipeline: single nightly cron
One `player-db-cron` Railway service runs ~2am ET. Steps in order:
1. Fetch + upsert yesterday's game logs
2. Patch team schedule postponements
3. Recompute season aggregates (existing `PlayerStatsManager`)
4. Compute std dev from game logs → write back to aggregate tables

Opponent defense rankings on separate weekly cadence (Monday morning).

Uses existing `is_nba_season()` skip logic for July–September.

## What's Next

Design not yet complete — stopped after Section 2 (Pipeline). Need to:
1. Present Section 3 (Agent tools) and get approval
2. Write the design doc to `docs/superpowers/specs/`
3. Run spec self-review
4. Get user spec review
5. Invoke `writing-plans` to create implementation plan
