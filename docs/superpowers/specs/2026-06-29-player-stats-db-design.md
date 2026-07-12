# Spec: Player Stats Database Layer

**Date:** 2026-06-29
**Branch:** feature/waiver_wire (implementation will be on a new branch)
**Status:** Approved — ready for implementation planning

---

## Goal

Replace live NBA API calls for in-season stats queries with a persistent Postgres database. Agents query the DB; a nightly cron keeps it fresh. Enables trend analysis, schedule density queries, and custom SQL — none of which are practical with live API calls due to rate limiting (~1s+ per call).

---

## Section 1: Schema

Six tables total: three existing aggregate tables (schema additions only) and three new tables.

### Existing tables (no schema changes)

| Table | Scope | Key |
|-------|-------|-----|
| `bball_monsters_player_stats_pg` | All seasons, per-game | `(player_id, season)` |
| `bball_monsters_player_stats_total` | All seasons, totals | `(player_id, season)` |
| `bball_monsters_player_stats_p36` | All seasons, per-36 | `(player_id, season)` |

These are managed by `data/postgres/player_stats.py` → `PlayerStatsManager`. **Add `std_dev_pts`, `std_dev_reb`, `std_dev_ast`, `std_dev_stl`, `std_dev_blk`, `std_dev_to`, `std_dev_fgp`, `std_dev_3pp`, `std_dev_ftp` columns** (FLOAT, nullable) to each. Populated daily from game logs; preserved at season end after game logs are truncated.

### New tables

#### `player_game_logs`

Current season only. Truncated at season end (std dev already written back to aggregates).

```sql
CREATE TABLE IF NOT EXISTS player_game_logs (
    player_id         INTEGER      NOT NULL,
    game_id           VARCHAR(20)  NOT NULL,
    game_date         DATE         NOT NULL,
    season            VARCHAR(10)  NOT NULL,
    team_abbreviation VARCHAR(10),
    matchup           VARCHAR(20),
    wl                CHAR(1),
    min               FLOAT,
    pts               FLOAT,
    reb               FLOAT,
    ast               FLOAT,
    stl               FLOAT,
    blk               FLOAT,
    "to"              FLOAT,
    fgm               FLOAT,
    fga               FLOAT,
    fg_pct            FLOAT,
    fg3m              FLOAT,
    fg3a              FLOAT,
    fg3_pct           FLOAT,
    ftm               FLOAT,
    fta               FLOAT,
    ft_pct            FLOAT,
    plus_minus        FLOAT,
    updated_at        TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (player_id, game_id)
);
```

#### `team_schedules`

Current season. Updated once at season start + postponement patches.

```sql
CREATE TABLE IF NOT EXISTS team_schedules (
    team_id           INTEGER      NOT NULL,
    team_abbreviation VARCHAR(10)  NOT NULL,
    game_id           VARCHAR(20)  NOT NULL,
    game_date         DATE         NOT NULL,
    season            VARCHAR(10)  NOT NULL,
    home_away         VARCHAR(10),
    opponent_abbr     VARCHAR(10),
    updated_at        TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (team_id, game_id)
);
```

#### `opponent_defense_rankings`

Weekly snapshot (Monday mornings). Enables matchup-quality analysis.

```sql
CREATE TABLE IF NOT EXISTS opponent_defense_rankings (
    team_id           INTEGER      NOT NULL,
    team_abbreviation VARCHAR(10)  NOT NULL,
    season            VARCHAR(10)  NOT NULL,
    rank_pts          INTEGER,
    rank_reb          INTEGER,
    rank_ast          INTEGER,
    rank_stl          INTEGER,
    rank_blk          INTEGER,
    rank_to           INTEGER,
    rank_fg_pct       INTEGER,
    rank_3p_pct       INTEGER,
    updated_at        TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (team_id, season)
);
```

### Explicitly excluded

- Historical game logs — season aggregates cover dynasty needs
- Player injury/status — live from Yahoo
- League roster/ownership — live from Yahoo

---

## Section 2: Pipeline

### Service: `player-db-cron` (new Railway service)

Schedule: `0 6 * * *` (2am ET = 6am UTC). Skips July–September via existing `is_nba_season()`.

### Daily run order

1. **Fetch + upsert yesterday's game logs** — `PlayerGameLogUpdater`
   - Call NBA API `PlayerGameLog` for each active player OR `LeagueGameFinder` for yesterday's games (more efficient — one call gets all players)
   - Upsert into `player_game_logs`

2. **Patch team schedule postponements** — `TeamScheduleUpdater`
   - Check yesterday's scheduled games; mark any that didn't occur
   - Full schedule loaded once at season start (October)

3. **Recompute season aggregates** — existing `PlayerStatsManager.fetch()` + `upsert_all()`
   - 3 NBA API calls (PerGame, Totals, Per36); upserts all active players

4. **Compute std dev → write back to aggregates** — `StdDevUpdater`
   - `SELECT player_id, STDDEV(pts), STDDEV(reb), ...` grouped from `player_game_logs`
   - UPDATE `bball_monsters_player_stats_pg/total/p36` std_dev columns

### Weekly (Monday): opponent defense rankings

5. **Fetch + upsert opponent defense rankings** — `OpponentDefenseUpdater`
   - NBA API `LeagueDashTeamStats` in opponent mode
   - Rank teams 1–30 per stat; upsert into `opponent_defense_rankings`

### Entry point

`scripts/pipelines/update_player_stats_db.py` — mirrors structure of `update_vector_database.py`. Runs all daily steps; Monday check runs weekly step too.

---

## Section 3: Agent Tools

Four tools added to **StatsAgent** (`agents/tools/player_stats_db.py`, new file):

### Pre-built tools

**`get_recent_form(player_id, n_games=10)`**
- Queries `player_game_logs` for last N games
- Returns: per-game averages (PTS, REB, AST, STL, BLK, TOV, FG%, 3P%, FT%) + delta vs. season average
- Use cases: hot/cold streak detection, waiver wire form check

**`get_schedule_density(team_abbreviation, days=7)`**
- Queries `team_schedules` for games in next N days
- Returns: list of games (date, opponent, home/away) + total count + avg days rest
- Use cases: streaming pickup recommendations, start/sit decisions

**`get_season_trends(player_id)`**
- Queries `player_game_logs` aggregated by rolling 30-day windows
- Returns: monthly averages for key cats showing trajectory over the season
- Use cases: buy-low/sell-high analysis, production trajectory

### Raw SQL fallback

**`query_stats_db(sql)`**
- Executes a read-only SELECT against the stats tables
- Enforces SELECT-only at the tool level (rejects INSERT/UPDATE/DELETE/DROP)
- Returns: JSON rows
- Use cases: custom filters, cross-table joins, opponent matchup queries the agent writes itself

### StatsAgent routing guidance (system prompt addition)

```
For in-season stats analysis:
- Use get_recent_form for hot/cold streak and recent performance queries
- Use get_schedule_density for games-remaining and matchup-density questions
- Use get_season_trends for trajectory and buy/sell analysis
- Use query_stats_db for anything custom — cross-table queries, specific filters
- Fall back to live NBA API tools only for data not in the DB (live box scores, play-by-play)
```

### Tool file location

`agents/tools/player_stats_db.py` — new file, follows pattern of `agents/tools/base.py`.

---

## Open Questions

None — all design decisions made and approved.

---

## Implementation Notes

- `PlayerStatsManager` in `data/postgres/player_stats.py` already handles the 3 aggregate tables; the new pipeline script extends it, doesn't replace it
- All new DB classes should live in `data/postgres/` following existing patterns
- New Railway service (`player-db-cron`) needs `DATABASE_URL` and `OPENAI_API_KEY` env vars (inherits from Railway project)
- `is_nba_season()` and `current_season()` helpers already in `data/postgres/player_stats.py` — reuse them
