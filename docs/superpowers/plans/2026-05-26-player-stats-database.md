# NBA Player Stats Database Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create 3 NBA player stats tables (per-game, totals, per-36), initialize with last 3 seasons, compute z-scores and value metrics (rV, pV, three_v) per player per season, run a daily Railway cron, and expose a `basketball_monster_stats` LangChain tool to agents.

**Architecture:**
- `PlayerStatsManager` (`data/postgres/player_stats.py`) — 3 NBA API calls per season (PerGame, Totals, Per36), upserts into 3 tables
- `PlayerValueCalculator` (`data/postgres/player_value.py`) — reads from all 3 tables, computes 9 z-scores + aggregate metrics per context, writes back
- `scripts/pipelines/init_player_stats.py` — one-time: last 3 seasons
- `scripts/pipelines/update_player_stats.py` — daily cron: current season, skips July–September
- `PlayerStatsTool` (`agents/tools/player_stats.py`) — `basketball_monster_stats` LangChain tool

**Tech Stack:** `nba_api` (`LeagueDashPlayerStats` × 3 modes), `pandas`/`numpy` (z-scores), SQLAlchemy + psycopg2, Railway PostgreSQL, LangChain `@tool`

---

## Table Schemas

**Shared identity columns** (all 3 tables):
`player_id INTEGER, season VARCHAR(10), player_name, team_abbreviation, age FLOAT, gp INTEGER`
`PRIMARY KEY (player_id, season)`

**Shared z-score columns** (all 3 tables, computed from each table's own stats):
`z_pts, z_reb, z_ast, z_stl, z_blk, z_3ptm, z_tov, z_fg, z_ft`
- `z_3ptm` = z-score of `fg3m` (3-pointers made) → also stored as `three_v`
- `z_fg` = volume-weighted FG% z-score: `(fg_pct - mean_fg_pct) × fga`, then z-score
- `z_ft` = volume-weighted FT% z-score: `(ft_pct - mean_ft_pct) × fta`, then z-score
- `z_tov` = negated: `-(tov - mean_tov) / std_tov`
- Player pool for all z-scores: top 150 by `min` from `bball_monsters_player_stats_pg`

**`bball_monsters_player_stats_pg`**
```
min, pts, reb, ast, stl, blk, tov,
fgm, fga, fg_pct, fg3m, fg3a, fg3_pct,
ftm, fta, ft_pct, oreb, dreb, pf, plus_minus,
[9 z-scores],
rv, three_v, pv
```

**`bball_monsters_player_stats_total`**
```
min, pts, reb, ast, stl, blk, tov,
fgm, fga, fg_pct, fg3m, fg3a, fg3_pct,
ftm, fta, ft_pct, oreb, dreb, pf, plus_minus,
[9 z-scores],
rv, three_v          ← no pv (totals ≠ per-game points league value)
```

**`bball_monsters_player_stats_p36`**
```
min,                  ← actual avg minutes played (context for the rates)
pts, reb, ast, stl, blk, tov,
fgm, fga, fg_pct, fg3m, fg3a, fg3_pct,
ftm, fta, ft_pct, oreb, dreb, pf,
[9 z-scores],
rv, three_v, pv
```

**pV formula (Yahoo points league):**
`raw_pv = pts×1.0 + reb×1.2 + ast×1.5 + stl×3.0 + blk×3.0 + tov×(-1.0)`
`pv = raw_pv - raw_pv_of_150th_player`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `data/postgres/player_stats.py` | `PlayerStatsManager` — 3-table DDL, 3-mode NBA API fetch, upsert |
| Create | `data/postgres/player_value.py` | `PlayerValueCalculator` — z-scores + rv/pv/three_v for all 3 tables |
| Create | `scripts/pipelines/update_player_stats.py` | Daily cron — in-season guard, fetch → upsert → calculate |
| Create | `scripts/pipelines/init_player_stats.py` | One-time init — last 3 seasons |
| Create | `agents/tools/player_stats.py` | `PlayerStatsTool` — `basketball_monster_stats` LangChain tool |
| Modify | `agents/tools/__init__.py` | Export `PlayerStatsTool` |
| Modify | `agents/multi_agent/statistic_agent.py` | Load `PlayerStatsTool` |
| Create | `tests/test_player_stats_manager.py` | Manager unit tests |
| Create | `tests/test_player_value_calculator.py` | Calculator unit tests |
| Create | `tests/test_player_stats_tool.py` | Tool unit tests |

---

## Task 1: `PlayerStatsManager` — `data/postgres/player_stats.py`

Creates all 3 tables, fetches 3 NBA API modes, upserts into all 3 tables. Season utilities live here too.

**Files:**
- Create: `data/postgres/player_stats.py`
- Create: `tests/test_player_stats_manager.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_player_stats_manager.py`:

```python
import pandas as pd
from datetime import date
from unittest.mock import MagicMock, patch

_API_ROW = {
    'PLAYER_ID': 2544, 'PLAYER_NAME': 'LeBron James',
    'TEAM_ABBREVIATION': 'LAL', 'AGE': 40.0, 'GP': 70,
    'MIN': 35.5, 'PTS': 25.7, 'REB': 7.4, 'AST': 8.3,
    'STL': 1.3, 'BLK': 0.5, 'TOV': 3.5,
    'FGM': 9.2, 'FGA': 18.0, 'FG_PCT': 0.511,
    'FG3M': 2.1, 'FG3A': 5.8, 'FG3_PCT': 0.362,
    'FTM': 5.0, 'FTA': 6.8, 'FT_PCT': 0.735,
    'OREB': 1.3, 'DREB': 6.1, 'PF': 1.4, 'PLUS_MINUS': 3.2,
}


def _make_manager():
    mock_conn = MagicMock()
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_pg = MagicMock()
    mock_pg.get_engine.return_value = mock_engine
    from data.postgres.player_stats import PlayerStatsManager
    return PlayerStatsManager(mock_pg), mock_conn


# --- create_tables ---

def test_create_tables_returns_true():
    mgr, _ = _make_manager()
    assert mgr.create_tables() is True

def test_create_tables_creates_pg_table():
    mgr, mock_conn = _make_manager()
    mgr.create_tables()
    sqls = [call[0][0].text for call in mock_conn.execute.call_args_list]
    assert any('bball_monsters_player_stats_pg' in s for s in sqls)

def test_create_tables_creates_total_table():
    mgr, mock_conn = _make_manager()
    mgr.create_tables()
    sqls = [call[0][0].text for call in mock_conn.execute.call_args_list]
    assert any('bball_monsters_player_stats_total' in s for s in sqls)

def test_create_tables_creates_p36_table():
    mgr, mock_conn = _make_manager()
    mgr.create_tables()
    sqls = [call[0][0].text for call in mock_conn.execute.call_args_list]
    assert any('bball_monsters_player_stats_p36' in s for s in sqls)

def test_create_tables_returns_false_on_error():
    mock_pg = MagicMock()
    mock_pg.get_engine.side_effect = Exception("DB down")
    from data.postgres.player_stats import PlayerStatsManager
    assert PlayerStatsManager(mock_pg).create_tables() is False


# --- fetch ---

def test_fetch_makes_three_api_calls():
    endpoints = [MagicMock(), MagicMock(), MagicMock()]
    for ep in endpoints:
        ep.get_data_frames.return_value = [pd.DataFrame([_API_ROW])]
    with patch('data.postgres.player_stats.LeagueDashPlayerStats', side_effect=endpoints):
        from data.postgres.player_stats import PlayerStatsManager
        pg, tot, p36 = PlayerStatsManager(MagicMock()).fetch('2025-26')
    assert len(pg) == len(tot) == len(p36) == 1

def test_fetch_returns_three_dataframes():
    endpoints = [MagicMock(), MagicMock(), MagicMock()]
    for ep in endpoints:
        ep.get_data_frames.return_value = [pd.DataFrame([_API_ROW])]
    with patch('data.postgres.player_stats.LeagueDashPlayerStats', side_effect=endpoints):
        from data.postgres.player_stats import PlayerStatsManager
        result = PlayerStatsManager(MagicMock()).fetch('2025-26')
    assert len(result) == 3


# --- upsert ---

def test_upsert_pg_executes_for_each_row():
    mgr, mock_conn = _make_manager()
    df = pd.DataFrame([_API_ROW])
    mgr.upsert_pg(df, '2025-26')
    assert mock_conn.execute.call_count == 1

def test_upsert_pg_sql_has_on_conflict():
    mgr, mock_conn = _make_manager()
    mgr.upsert_pg(pd.DataFrame([_API_ROW]), '2025-26')
    sql = mock_conn.execute.call_args[0][0].text
    assert 'ON CONFLICT' in sql
    assert 'bball_monsters_player_stats_pg' in sql

def test_upsert_total_sql_targets_total_table():
    mgr, mock_conn = _make_manager()
    mgr.upsert_total(pd.DataFrame([_API_ROW]), '2025-26')
    sql = mock_conn.execute.call_args[0][0].text
    assert 'bball_monsters_player_stats_total' in sql

def test_upsert_p36_sql_targets_p36_table():
    mgr, mock_conn = _make_manager()
    mgr.upsert_p36(pd.DataFrame([_API_ROW]), '2025-26')
    sql = mock_conn.execute.call_args[0][0].text
    assert 'bball_monsters_player_stats_p36' in sql

def test_upsert_maps_player_columns():
    mgr, mock_conn = _make_manager()
    mgr.upsert_pg(pd.DataFrame([_API_ROW]), '2025-26')
    params = mock_conn.execute.call_args[0][1]
    assert params['player_id'] == 2544
    assert params['player_name'] == 'LeBron James'
    assert params['pts'] == 25.7
    assert params['fg3m'] == 2.1

def test_upsert_commits():
    mgr, mock_conn = _make_manager()
    mgr.upsert_pg(pd.DataFrame([_API_ROW, _API_ROW]), '2025-26')
    mock_conn.commit.assert_called_once()


# --- season utilities ---

def test_is_nba_season_true_in_october():
    from data.postgres.player_stats import is_nba_season
    assert is_nba_season(date(2025, 10, 15)) is True

def test_is_nba_season_true_in_june():
    from data.postgres.player_stats import is_nba_season
    assert is_nba_season(date(2026, 6, 10)) is True

def test_is_nba_season_false_in_july():
    from data.postgres.player_stats import is_nba_season
    assert is_nba_season(date(2026, 7, 1)) is False

def test_is_nba_season_false_in_august():
    from data.postgres.player_stats import is_nba_season
    assert is_nba_season(date(2026, 8, 15)) is False

def test_is_nba_season_false_in_september():
    from data.postgres.player_stats import is_nba_season
    assert is_nba_season(date(2026, 9, 30)) is False

def test_current_season_october():
    from data.postgres.player_stats import current_season
    assert current_season(date(2025, 10, 20)) == '2025-26'

def test_current_season_january():
    from data.postgres.player_stats import current_season
    assert current_season(date(2026, 1, 15)) == '2025-26'

def test_last_n_seasons_returns_correct_list():
    from data.postgres.player_stats import last_n_seasons
    assert last_n_seasons(date(2026, 5, 27), n=3) == ['2023-24', '2024-25', '2025-26']
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_player_stats_manager.py -v
```

Expected: `ModuleNotFoundError: No module named 'data.postgres.player_stats'`

- [ ] **Step 3: Implement `data/postgres/player_stats.py`**

```python
"""
PlayerStatsManager — NBA stats across 3 tables (per-game, totals, per-36).

Three NBA API calls per season → 3 upserts.
Season utilities (is_nba_season, current_season, last_n_seasons) are module-level.
"""

from datetime import date
from typing import Optional

import pandas as pd
from sqlalchemy import text

from nba_api.stats.endpoints import leaguedashplayerstats as _ep
LeagueDashPlayerStats = _ep.LeagueDashPlayerStats

from data.postgres.connection import PostgresConnection
from logger import get_logger

logger = get_logger(__name__)

_STAT_COLS = [
    'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV',
    'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
    'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'PF', 'PLUS_MINUS',
]
_P36_STAT_COLS = [c for c in _STAT_COLS if c != 'PLUS_MINUS']  # no plus_minus in per-36
_INFO_COLS = ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'AGE', 'GP']

_Z_COLS = ['z_pts', 'z_reb', 'z_ast', 'z_stl', 'z_blk', 'z_3ptm',
           'z_tov', 'z_fg', 'z_ft']
_PG_VALUE_COLS  = ['rv', 'three_v', 'pv']
_TOT_VALUE_COLS = ['rv', 'three_v']
_P36_VALUE_COLS = ['rv', 'three_v', 'pv']


def _col_list(cols):
    return ', '.join(c.lower() for c in cols)

def _param_list(cols):
    return ', '.join(f':{c.lower()}' for c in cols)

def _update_list(cols):
    c = [c.lower() for c in cols]
    return ', '.join(f'{col} = EXCLUDED.{col}' for col in c)

def _null_value_cols(cols):
    return ', '.join(f'NULL AS {c}' for c in cols)


def _make_create_sql(table: str, stat_cols: list, value_cols: list) -> text:
    stat_ddl  = ',\n        '.join(f'{c.lower()} FLOAT' for c in stat_cols)
    z_ddl     = ',\n        '.join(f'{c} FLOAT' for c in _Z_COLS)
    value_ddl = ',\n        '.join(f'{c} FLOAT' for c in value_cols)
    return text(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            player_id         INTEGER      NOT NULL,
            season            VARCHAR(10)  NOT NULL,
            player_name       VARCHAR(100),
            team_abbreviation VARCHAR(10),
            age               FLOAT,
            gp                INTEGER,
            {stat_ddl},
            {z_ddl},
            {value_ddl},
            updated_at        TIMESTAMP DEFAULT NOW(),
            PRIMARY KEY (player_id, season)
        )
    """)


def _make_upsert_sql(table: str, stat_cols: list, value_cols: list) -> text:
    all_stat  = [c.lower() for c in stat_cols]
    all_value = list(value_cols)
    all_z     = list(_Z_COLS)

    insert_cols   = ['player_id', 'season', 'player_name', 'team_abbreviation', 'age', 'gp'] + all_stat
    insert_params = [f':{c}' for c in insert_cols]
    update_cols   = ['player_name', 'team_abbreviation', 'age', 'gp'] + all_stat + all_z + all_value

    return text(f"""
        INSERT INTO {table} (
            {', '.join(insert_cols)},
            {', '.join(all_z)},
            {', '.join(all_value)},
            updated_at
        ) VALUES (
            {', '.join(insert_params)},
            {', '.join(f':{c}' for c in all_z)},
            {', '.join(f':{c}' for c in all_value)},
            NOW()
        )
        ON CONFLICT (player_id, season) DO UPDATE SET
            {', '.join(f'{c} = EXCLUDED.{c}' for c in update_cols)},
            updated_at = NOW()
    """)


_PG_TABLE    = 'bball_monsters_player_stats_pg'
_TOT_TABLE   = 'bball_monsters_player_stats_total'
_P36_TABLE   = 'bball_monsters_player_stats_p36'

_CREATE_PG    = _make_create_sql(_PG_TABLE,  _STAT_COLS,     _PG_VALUE_COLS)
_CREATE_TOT   = _make_create_sql(_TOT_TABLE, _STAT_COLS,     _TOT_VALUE_COLS)
_CREATE_P36   = _make_create_sql(_P36_TABLE, _P36_STAT_COLS, _P36_VALUE_COLS)

_UPSERT_PG    = _make_upsert_sql(_PG_TABLE,  _STAT_COLS,     _PG_VALUE_COLS)
_UPSERT_TOT   = _make_upsert_sql(_TOT_TABLE, _STAT_COLS,     _TOT_VALUE_COLS)
_UPSERT_P36   = _make_upsert_sql(_P36_TABLE, _P36_STAT_COLS, _P36_VALUE_COLS)


# --- Season utilities ---

def is_nba_season(d: date = None) -> bool:
    """July, August, September are off-season."""
    d = d or date.today()
    return d.month not in (7, 8, 9)


def current_season(d: date = None) -> str:
    d = d or date.today()
    year = d.year if d.month >= 10 else d.year - 1
    return f"{year}-{str(year + 1)[2:]}"


def last_n_seasons(d: date = None, n: int = 3) -> list:
    d = d or date.today()
    end_year = int(d.year if d.month >= 10 else d.year - 1)
    return [f"{end_year - (n - 1 - i)}-{str(end_year - (n - 2 - i))[2:]}" for i in range(n)]


# --- Manager ---

class PlayerStatsManager:
    """Creates 3 stats tables, fetches from 3 NBA API modes, upserts all 3."""

    def __init__(self, connection: PostgresConnection):
        self.connection = connection

    def create_tables(self) -> bool:
        try:
            engine = self.connection.get_engine()
            with engine.connect() as conn:
                for sql in [_CREATE_PG, _CREATE_TOT, _CREATE_P36]:
                    conn.execute(sql)
                conn.commit()
            logger.info("Created/verified all 3 player stats tables")
            return True
        except Exception as e:
            logger.error(f"Failed to create player stats tables: {e}")
            return False

    def _fetch_mode(self, season: str, mode: str) -> pd.DataFrame:
        return LeagueDashPlayerStats(
            season=season, per_mode_detailed=mode, timeout=30
        ).get_data_frames()[0]

    def fetch(self, season: str) -> tuple:
        """Returns (pg_df, tot_df, p36_df) from 3 NBA API calls."""
        logger.info(f"  Fetching PerGame...")
        pg  = self._fetch_mode(season, 'PerGame')
        logger.info(f"  Fetching Totals...")
        tot = self._fetch_mode(season, 'Totals')
        logger.info(f"  Fetching Per36...")
        p36 = self._fetch_mode(season, 'Per36')
        return pg, tot, p36

    def _upsert(self, sql: text, df: pd.DataFrame, season: str, stat_cols: list,
                value_cols: list) -> int:
        rows = df.to_dict('records')
        engine = self.connection.get_engine()
        with engine.connect() as conn:
            for row in rows:
                params = {
                    'player_id':         int(row['PLAYER_ID']),
                    'season':            season,
                    'player_name':       row['PLAYER_NAME'],
                    'team_abbreviation': row.get('TEAM_ABBREVIATION'),
                    'age':               row.get('AGE'),
                    'gp':                row.get('GP'),
                }
                for c in stat_cols:
                    params[c.lower()] = row.get(c)
                for c in _Z_COLS + value_cols:
                    params[c] = None  # filled in by PlayerValueCalculator
                conn.execute(sql, params)
            conn.commit()
        return len(rows)

    def upsert_pg(self, df: pd.DataFrame, season: str) -> int:
        return self._upsert(_UPSERT_PG, df, season, _STAT_COLS, _PG_VALUE_COLS)

    def upsert_total(self, df: pd.DataFrame, season: str) -> int:
        return self._upsert(_UPSERT_TOT, df, season, _STAT_COLS, _TOT_VALUE_COLS)

    def upsert_p36(self, df: pd.DataFrame, season: str) -> int:
        return self._upsert(_UPSERT_P36, df, season, _P36_STAT_COLS, _P36_VALUE_COLS)

    def upsert_all(self, pg: pd.DataFrame, tot: pd.DataFrame, p36: pd.DataFrame,
                   season: str) -> dict:
        return {
            'pg':  self.upsert_pg(pg, season),
            'tot': self.upsert_total(tot, season),
            'p36': self.upsert_p36(p36, season),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_player_stats_manager.py -v
```

Expected: 22 tests PASS

- [ ] **Step 5: Commit**

```bash
git add data/postgres/player_stats.py tests/test_player_stats_manager.py
git commit -m "feat: add PlayerStatsManager with 3-table schema and 3-mode NBA API fetch"
```

---

## Task 2: `PlayerValueCalculator` — `data/postgres/player_value.py`

Reads all 3 tables, computes z-scores (with volume-weighting for FG%/FT%), writes `z_*`, `rv`, `three_v`, `pv` back. Pool = top 150 players by `min` from the PG table.

**Files:**
- Create: `data/postgres/player_value.py`
- Create: `tests/test_player_value_calculator.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_player_value_calculator.py`:

```python
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch, call

POOL_SIZE = 150

def _make_pool_df(n=160):
    """Create a fake player pool DataFrame with realistic variance."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'player_id':   range(n),
        'season':      ['2025-26'] * n,
        'min':         rng.uniform(15, 38, n),
        'pts':         rng.uniform(5, 32, n),
        'reb':         rng.uniform(2, 14, n),
        'ast':         rng.uniform(1, 12, n),
        'stl':         rng.uniform(0.3, 2.5, n),
        'blk':         rng.uniform(0.1, 3.0, n),
        'tov':         rng.uniform(0.8, 4.5, n),
        'fg3m':        rng.uniform(0, 4.5, n),
        'fg_pct':      rng.uniform(0.38, 0.65, n),
        'fga':         rng.uniform(3, 20, n),
        'ft_pct':      rng.uniform(0.60, 0.95, n),
        'fta':         rng.uniform(0.5, 9, n),
    })


def _make_calculator():
    mock_conn = MagicMock()
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_pg = MagicMock()
    mock_pg.get_engine.return_value = mock_engine
    from data.postgres.player_value import PlayerValueCalculator
    return PlayerValueCalculator(mock_pg), mock_conn


# --- z-score functions ---

def test_z_score_of_mean_is_zero():
    from data.postgres.player_value import z_score
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = z_score(s)
    assert abs(result[2]) < 1e-10  # middle value (mean) ≈ 0

def test_z_score_negated_flips_sign():
    from data.postgres.player_value import z_score
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    pos = z_score(s)
    neg = z_score(s, negate=True)
    pd.testing.assert_series_equal(pos, -neg)

def test_z_score_volume_weighted_fg():
    from data.postgres.player_value import z_score_volume_weighted
    pct   = pd.Series([0.50, 0.45, 0.55])
    vol   = pd.Series([10.0, 5.0, 20.0])
    result = z_score_volume_weighted(pct, vol)
    assert isinstance(result, pd.Series)
    assert len(result) == 3


# --- pool selection ---

def test_pool_is_top_150_by_min():
    from data.postgres.player_value import select_pool
    df = _make_pool_df(n=200)
    pool = select_pool(df, size=150)
    assert len(pool) == 150
    assert pool['min'].min() >= df.nlargest(150, 'min')['min'].min()


# --- calculate ---

def test_calculate_writes_z_scores_back_to_pg():
    calc, mock_conn = _make_calculator()
    df = _make_pool_df(n=160)
    mock_result = MagicMock()
    mock_result.fetchall.return_value = [tuple(row) for _, row in df.iterrows()]
    mock_result.keys.return_value = list(df.columns)
    mock_conn.execute.return_value = mock_result

    with patch.object(calc, '_read_table', return_value=df):
        with patch.object(calc, '_write_values') as mock_write:
            calc.calculate('2025-26', table='pg')
            mock_write.assert_called_once()
            written_df = mock_write.call_args[0][0]
            assert 'z_pts' in written_df.columns
            assert 'z_fg' in written_df.columns
            assert 'z_3ptm' in written_df.columns
            assert 'rv' in written_df.columns

def test_calculate_rv_is_sum_of_nine_z_scores():
    from data.postgres.player_value import PlayerValueCalculator
    calc = PlayerValueCalculator(MagicMock())
    df = _make_pool_df(n=160)
    result = calc._compute_values(df, include_pv=True)
    z_cols = ['z_pts', 'z_reb', 'z_ast', 'z_stl', 'z_blk',
              'z_3ptm', 'z_tov', 'z_fg', 'z_ft']
    expected_rv = result[z_cols].sum(axis=1)
    pd.testing.assert_series_equal(result['rv'], expected_rv, check_names=False)

def test_calculate_three_v_equals_z_3ptm():
    from data.postgres.player_value import PlayerValueCalculator
    calc = PlayerValueCalculator(MagicMock())
    df = _make_pool_df(n=160)
    result = calc._compute_values(df, include_pv=True)
    pd.testing.assert_series_equal(result['three_v'], result['z_3ptm'], check_names=False)

def test_calculate_pv_replacement_is_150th():
    from data.postgres.player_value import PlayerValueCalculator
    calc = PlayerValueCalculator(MagicMock())
    df = _make_pool_df(n=160)
    result = calc._compute_values(df, include_pv=True)
    # Player ranked exactly 150th by raw_pv should have pv ≈ 0
    raw_pv = (df['pts'] * 1.0 + df['reb'] * 1.2 + df['ast'] * 1.5
              + df['stl'] * 3.0 + df['blk'] * 3.0 + df['tov'] * -1.0)
    replacement = raw_pv.nlargest(150).iloc[-1]
    assert abs((raw_pv - replacement).iloc[0] - result['pv'].iloc[0]) < 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_player_value_calculator.py -v
```

Expected: `ModuleNotFoundError: No module named 'data.postgres.player_value'`

- [ ] **Step 3: Implement `data/postgres/player_value.py`**

```python
"""
PlayerValueCalculator — computes z-scores and value metrics for all 3 stat tables.

Pool: top 150 players by `min` from bball_monsters_player_stats_pg.
FG% and FT% z-scores are volume-weighted.
Writes z_pts, z_reb, z_ast, z_stl, z_blk, z_3ptm, z_tov, z_fg, z_ft,
rv, three_v (all tables), and pv (pg + p36 tables) back in-place.
"""

import numpy as np
import pandas as pd
from sqlalchemy import text

from data.postgres.connection import PostgresConnection
from logger import get_logger

logger = get_logger(__name__)

POOL_SIZE = 150

_PG_TABLE  = 'bball_monsters_player_stats_pg'
_TOT_TABLE = 'bball_monsters_player_stats_total'
_P36_TABLE = 'bball_monsters_player_stats_p36'

_PV_WEIGHTS = {'pts': 1.0, 'reb': 1.2, 'ast': 1.5,
               'stl': 3.0, 'blk': 3.0, 'tov': -1.0}

_READ_SQL = lambda table, season: text(f"""
    SELECT player_id, season, min, pts, reb, ast, stl, blk, tov,
           fg3m, fg_pct, fga, ft_pct, fta
    FROM {table}
    WHERE season = :season
""")

_WRITE_SQL = lambda table: text(f"""
    UPDATE {table}
    SET z_pts   = :z_pts,   z_reb   = :z_reb,   z_ast   = :z_ast,
        z_stl   = :z_stl,   z_blk   = :z_blk,   z_3ptm  = :z_3ptm,
        z_tov   = :z_tov,   z_fg    = :z_fg,     z_ft    = :z_ft,
        rv      = :rv,      three_v = :three_v,
        updated_at = NOW()
    WHERE player_id = :player_id AND season = :season
""")

_WRITE_WITH_PV_SQL = lambda table: text(f"""
    UPDATE {table}
    SET z_pts   = :z_pts,   z_reb   = :z_reb,   z_ast   = :z_ast,
        z_stl   = :z_stl,   z_blk   = :z_blk,   z_3ptm  = :z_3ptm,
        z_tov   = :z_tov,   z_fg    = :z_fg,     z_ft    = :z_ft,
        rv      = :rv,      three_v = :three_v,  pv      = :pv,
        updated_at = NOW()
    WHERE player_id = :player_id AND season = :season
""")


def z_score(s: pd.Series, negate: bool = False) -> pd.Series:
    """Standard z-score. negate=True for TOV (lower is better)."""
    result = (s - s.mean()) / s.std(ddof=0)
    return -result if negate else result


def z_score_volume_weighted(pct: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Volume-weighted percentage z-score.
    impact = (player_pct - mean_pct) * player_volume
    then standard z-score of impact.
    """
    impact = (pct - pct.mean()) * volume
    return z_score(impact)


def select_pool(df: pd.DataFrame, size: int = POOL_SIZE) -> pd.DataFrame:
    """Top N players by minutes played."""
    return df.nlargest(size, 'min').reset_index(drop=True)


class PlayerValueCalculator:
    """Computes and writes z-scores + value metrics to all 3 player stats tables."""

    def __init__(self, connection: PostgresConnection):
        self.connection = connection

    def _read_table(self, table: str, season: str) -> pd.DataFrame:
        engine = self.connection.get_engine()
        with engine.connect() as conn:
            result = conn.execute(_READ_SQL(table, season), {'season': season})
            rows = result.fetchall()
            keys = list(result.keys())
        return pd.DataFrame([dict(zip(keys, row)) for row in rows])

    def _compute_values(self, df: pd.DataFrame, include_pv: bool) -> pd.DataFrame:
        df = df.copy()
        pool = select_pool(df)

        # Compute z-scores on pool, apply to all players
        def _z(col, negate=False):
            mean, std = pool[col].mean(), pool[col].std(ddof=0)
            result = (df[col] - mean) / std
            return -result if negate else result

        def _z_vw(pct_col, vol_col):
            mean_pct = pool[pct_col].mean()
            impact_pool = (pool[pct_col] - mean_pct) * pool[vol_col]
            mean_i, std_i = impact_pool.mean(), impact_pool.std(ddof=0)
            impact_all = (df[pct_col] - mean_pct) * df[vol_col]
            return (impact_all - mean_i) / std_i

        df['z_pts']   = _z('pts')
        df['z_reb']   = _z('reb')
        df['z_ast']   = _z('ast')
        df['z_stl']   = _z('stl')
        df['z_blk']   = _z('blk')
        df['z_3ptm']  = _z('fg3m')
        df['z_tov']   = _z('tov', negate=True)
        df['z_fg']    = _z_vw('fg_pct', 'fga')
        df['z_ft']    = _z_vw('ft_pct', 'fta')

        z_cols = ['z_pts', 'z_reb', 'z_ast', 'z_stl', 'z_blk',
                  'z_3ptm', 'z_tov', 'z_fg', 'z_ft']
        df['rv']      = df[z_cols].sum(axis=1)
        df['three_v'] = df['z_3ptm']

        if include_pv:
            raw_pv = sum(df[stat] * w for stat, w in _PV_WEIGHTS.items())
            replacement = raw_pv.nlargest(POOL_SIZE).iloc[-1]
            df['pv'] = raw_pv - replacement

        return df

    def _write_values(self, df: pd.DataFrame, table: str, include_pv: bool):
        sql = _WRITE_WITH_PV_SQL(table) if include_pv else _WRITE_SQL(table)
        engine = self.connection.get_engine()
        with engine.connect() as conn:
            for row in df.to_dict('records'):
                params = {
                    'player_id': row['player_id'], 'season': row['season'],
                    'z_pts':  row['z_pts'],  'z_reb': row['z_reb'],
                    'z_ast':  row['z_ast'],  'z_stl': row['z_stl'],
                    'z_blk':  row['z_blk'],  'z_3ptm': row['z_3ptm'],
                    'z_tov':  row['z_tov'],  'z_fg':  row['z_fg'],
                    'z_ft':   row['z_ft'],   'rv':    row['rv'],
                    'three_v': row['three_v'],
                }
                if include_pv:
                    params['pv'] = row['pv']
                conn.execute(sql, params)
            conn.commit()

    def calculate(self, season: str, table: str = 'all'):
        """
        Compute and write z-scores + value metrics for the given season.
        table: 'pg' | 'total' | 'p36' | 'all'
        """
        targets = {
            'pg':    (_PG_TABLE,  True),
            'total': (_TOT_TABLE, False),
            'p36':   (_P36_TABLE, True),
        }
        to_run = targets if table == 'all' else {table: targets[table]}

        # Pool is always defined by pg table min column
        pg_df = self._read_table(_PG_TABLE, season)

        for key, (tbl, include_pv) in to_run.items():
            df = pg_df if key == 'pg' else self._read_table(tbl, season)
            # Pass pg_df min column as pool anchor for non-pg tables
            if key != 'pg':
                df = df.merge(pg_df[['player_id', 'min']], on='player_id',
                              suffixes=('_orig', ''), how='left')
            result = self._compute_values(df, include_pv=include_pv)
            self._write_values(result, tbl, include_pv=include_pv)
            logger.info(f"  Calculated values for {tbl} ({len(result)} players)")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_player_value_calculator.py -v
```

Expected: 11 tests PASS

- [ ] **Step 5: Commit**

```bash
git add data/postgres/player_value.py tests/test_player_value_calculator.py
git commit -m "feat: add PlayerValueCalculator with z-scores, rv, three_v, pv"
```

---

## Task 3: Cron Script — `scripts/pipelines/update_player_stats.py`

**Files:**
- Create: `scripts/pipelines/update_player_stats.py`

- [ ] **Step 1: Implement**

```python
"""
Daily cron: refresh current-season NBA player stats + value metrics.
Exits early July–September (off-season).

Usage: python scripts/pipelines/update_player_stats.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.postgres.connection import PostgresConnection
from data.postgres.player_stats import PlayerStatsManager, is_nba_season, current_season
from data.postgres.player_value import PlayerValueCalculator
from logger import get_logger

logger = get_logger(__name__)


def main():
    if not is_nba_season():
        logger.info("Off-season (July–September) — skipping update")
        return

    season = current_season()
    pg_conn = PostgresConnection()
    mgr  = PlayerStatsManager(pg_conn)
    calc = PlayerValueCalculator(pg_conn)

    mgr.create_tables()

    logger.info(f"Fetching stats for {season}...")
    pg, tot, p36 = mgr.fetch(season)
    counts = mgr.upsert_all(pg, tot, p36, season)
    logger.info(f"Upserted — pg:{counts['pg']} tot:{counts['tot']} p36:{counts['p36']}")

    logger.info("Calculating value metrics...")
    calc.calculate(season, table='all')
    logger.info("Done")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Verify season utilities**

```bash
python -c "
from data.postgres.player_stats import is_nba_season, current_season
print('In season:', is_nba_season())
print('Current season:', current_season())
"
```

- [ ] **Step 3: Commit**

```bash
git add scripts/pipelines/update_player_stats.py
git commit -m "feat: add daily player stats cron script"
```

---

## Task 4: Init Script — `scripts/pipelines/init_player_stats.py`

**Files:**
- Create: `scripts/pipelines/init_player_stats.py`

- [ ] **Step 1: Implement**

```python
"""
One-time init: load last 3 NBA seasons into all 3 player stats tables.
Computes value metrics after each season.

Usage: python scripts/pipelines/init_player_stats.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.postgres.connection import PostgresConnection
from data.postgres.player_stats import PlayerStatsManager, last_n_seasons
from data.postgres.player_value import PlayerValueCalculator
from logger import get_logger

logger = get_logger(__name__)


def main():
    seasons = last_n_seasons(n=3)
    logger.info(f"Initializing player stats for seasons: {seasons}")

    pg_conn = PostgresConnection()
    mgr  = PlayerStatsManager(pg_conn)
    calc = PlayerValueCalculator(pg_conn)

    mgr.create_tables()

    for season in seasons:
        logger.info(f"Season {season}:")
        pg, tot, p36 = mgr.fetch(season)
        counts = mgr.upsert_all(pg, tot, p36, season)
        logger.info(f"  Upserted — pg:{counts['pg']} tot:{counts['tot']} p36:{counts['p36']}")
        calc.calculate(season, table='all')
        logger.info(f"  Value metrics calculated")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Verify season computation**

```bash
python -c "from data.postgres.player_stats import last_n_seasons; print(last_n_seasons(n=3))"
```

Expected: `['2023-24', '2024-25', '2025-26']`

- [ ] **Step 3: Commit**

```bash
git add scripts/pipelines/init_player_stats.py
git commit -m "feat: add one-time player stats init script"
```

---

## Task 5: Agent Query Tool — `agents/tools/player_stats.py`

**Files:**
- Create: `agents/tools/player_stats.py`
- Modify: `agents/tools/__init__.py`
- Create: `tests/test_player_stats_tool.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_player_stats_tool.py`:

```python
from unittest.mock import MagicMock, patch


def _make_tool(query_return=None):
    if query_return is None:
        query_return = [{'player_name': 'LeBron James', 'pts': 25.7,
                         'rv': 4.2, 'pv': 12.1, 'three_v': 1.3}]
    mock_conn = MagicMock()
    with patch('agents.tools.player_stats.PostgresConnection', return_value=mock_conn):
        from agents.tools.player_stats import PlayerStatsTool
        tool_obj = PlayerStatsTool(season='2025-26')
    tool_obj._query = MagicMock(return_value=query_return)
    return tool_obj


def test_tool_name_is_basketball_monster_stats():
    tool_obj = _make_tool()
    assert tool_obj.get_all_tools()[0].name == 'basketball_monster_stats'


def test_result_contains_player_and_values():
    tool_obj = _make_tool()
    result = tool_obj.get_all_tools()[0].invoke({'player_name': 'LeBron'})
    assert 'LeBron James' in result
    assert 'rv' in result
    assert 'pv' in result


def test_no_results_returns_message():
    tool_obj = _make_tool(query_return=[])
    result = tool_obj.get_all_tools()[0].invoke({'player_name': 'Nobody'})
    assert 'No players found' in result


def test_stat_type_defaults_to_pg():
    tool_obj = _make_tool()
    tool_obj.get_all_tools()[0].invoke({})
    call_args = tool_obj._query.call_args[1]
    assert call_args['stat_type'] == 'pg'


def test_stat_type_p36_is_passed():
    tool_obj = _make_tool()
    tool_obj.get_all_tools()[0].invoke({'stat_type': 'p36'})
    assert tool_obj._query.call_args[1]['stat_type'] == 'p36'


def test_season_override_is_passed():
    tool_obj = _make_tool()
    tool_obj.get_all_tools()[0].invoke({'season_override': '2023-24'})
    assert tool_obj._query.call_args[1]['season'] == '2023-24'
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_player_stats_tool.py -v
```

- [ ] **Step 3: Implement `agents/tools/player_stats.py`**

```python
"""
LangChain tool: basketball_monster_stats
Queries per-game, totals, or per-36 stats + z-scores + value metrics.
"""

import json
from typing import Optional

from langchain_core.tools import tool
from sqlalchemy import text

from agents.tools.base import ClutchAITool
from data.postgres.connection import PostgresConnection

_TABLES = {
    'pg':    'bball_monsters_player_stats_pg',
    'total': 'bball_monsters_player_stats_total',
    'p36':   'bball_monsters_player_stats_p36',
}


class PlayerStatsTool(ClutchAITool):
    """Query NBA player stats (pg/total/p36) with z-scores and value metrics."""

    def __init__(
        self,
        connection: Optional[PostgresConnection] = None,
        season: str = '2025-26',
        debug: bool = False,
    ):
        super().__init__(debug=debug)
        self.connection = connection or PostgresConnection()
        self.season = season

    def _query(
        self,
        season: str,
        stat_type: str = 'pg',
        player_name: Optional[str] = None,
        team: Optional[str] = None,
        limit: int = 20,
    ) -> list:
        table = _TABLES[stat_type]
        sql = text(f"""
            SELECT *
            FROM {table}
            WHERE season = :season
              AND (:player_name IS NULL OR player_name ILIKE :player_name)
              AND (:team IS NULL OR team_abbreviation = :team)
            ORDER BY rv DESC NULLS LAST
            LIMIT :limit
        """)
        engine = self.connection.get_engine()
        with engine.connect() as conn:
            result = conn.execute(sql, {
                'season':      season,
                'player_name': f'%{player_name}%' if player_name else None,
                'team':        team.upper() if team else None,
                'limit':       limit,
            })
            rows = result.fetchall()
            keys = list(result.keys())
        return [dict(zip(keys, row)) for row in rows]

    def create_basketball_monster_stats_tool(self):
        query_fn = self._query
        default_season = self.season

        @tool(
            "basketball_monster_stats",
            description=(
                "Query NBA player stats with z-scores and value metrics (Basketball Monster style). "
                "stat_type: 'pg' (per-game, default), 'total' (season totals), 'p36' (per-36 minutes). "
                "Returns stats, z_pts/z_reb/z_ast/z_stl/z_blk/z_3ptm/z_tov/z_fg/z_ft, "
                "rv (roto value), three_v (3-point value), pv (Yahoo points value). "
                "Filter by player_name (partial) or team abbreviation (e.g. 'LAL'). "
                "season_override: '2023-24', '2024-25', or '2025-26'."
            ),
        )
        def basketball_monster_stats(
            player_name: Optional[str] = None,
            team: Optional[str] = None,
            stat_type: str = 'pg',
            season_override: Optional[str] = None,
            limit: int = 20,
        ) -> str:
            season = season_override or default_season
            if stat_type not in _TABLES:
                return f"Invalid stat_type '{stat_type}'. Use: pg, total, p36."
            results = query_fn(
                season=season, stat_type=stat_type,
                player_name=player_name, team=team, limit=limit,
            )
            if not results:
                msg = f"No players found for season {season} [{stat_type}]"
                if player_name:
                    msg += f" matching '{player_name}'"
                if team:
                    msg += f" on team '{team}'"
                return msg
            return json.dumps(results, default=str)

        return basketball_monster_stats

    def get_all_tools(self):
        return [self.create_basketball_monster_stats_tool()]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_player_stats_tool.py -v
```

Expected: 6 tests PASS

- [ ] **Step 5: Update `agents/tools/__init__.py`**

```python
"""
ClutchAI tools package.
"""

from agents.tools.yahoo_api import YahooFantasyTool
from agents.tools.rotowire_rss import RotowireRSSFeedTool
from agents.tools.player_stats import PlayerStatsTool

__all__ = ['YahooFantasyTool', 'RotowireRSSFeedTool', 'PlayerStatsTool']
```

- [ ] **Step 6: Commit**

```bash
git add agents/tools/player_stats.py agents/tools/__init__.py tests/test_player_stats_tool.py
git commit -m "feat: add basketball_monster_stats LangChain tool"
```

---

## Task 6: Wire into `StatisticAgent`

**Files:**
- Modify: `agents/multi_agent/statistic_agent.py`

- [ ] **Step 1: Update `statistic_agent.py`**

Replace the entire file:

```python
"""
Statistic Agent — NBA stats and game data.

basketball_monster_stats: per-game/total/per-36 stats with z-scores (z_pts, z_reb,
z_ast, z_stl, z_blk, z_3ptm, z_tov, z_fg, z_ft) and value metrics (rv, pv, three_v).
NBA API tools: career stats, game logs, live scores, splits.
"""

from typing import List

from agents.multi_agent.base_agent import BaseAgent
from agents.tools.nba_api import nbaAPITool
from agents.tools.player_stats import PlayerStatsTool

logger = None


class StatisticAgent(BaseAgent):

    def _get_config_section(self) -> str:
        return 'statistic'

    def _get_default_system_prompt(self) -> str:
        return """You are a statistics specialist for fantasy basketball analysis.

Use basketball_monster_stats for player stats with value metrics:
- stat_type='pg'    → per-game averages + z-scores + rv, pv, three_v
- stat_type='total' → season totals + z-scores + rv, three_v
- stat_type='p36'   → per-36 minute rates + z-scores + rv, pv, three_v
Available seasons: 2023-24, 2024-25, 2025-26.

Z-scores: z_pts, z_reb, z_ast, z_stl, z_blk, z_3ptm (=three_v), z_tov, z_fg, z_ft.
rv = sum of all 9 z-scores. pv = Yahoo points league value vs replacement.

Use NBA API tools for career stats, game logs, live scores, and splits."""

    def _create_tools(self) -> List:
        tools = list(super()._create_base_tools())

        try:
            tools.extend(PlayerStatsTool(debug=self.debug).get_all_tools())
            self.logger.debug("PlayerStatsTool loaded")
        except Exception as e:
            self.logger.warning(f"PlayerStatsTool not available: {e}")

        try:
            tools.extend(nbaAPITool(debug=self.debug).get_all_tools())
            self.logger.debug("NBA API tools loaded")
        except Exception as e:
            self.logger.warning(f"NBA API tools not available: {e}")

        self.logger.info(f"Statistic Agent initialized with {len(tools)} tools")
        return tools
```

- [ ] **Step 2: Run all tests**

```bash
pytest tests/test_player_stats_manager.py tests/test_player_value_calculator.py tests/test_player_stats_tool.py -v
```

Expected: 39 tests PASS

- [ ] **Step 3: Commit**

```bash
git add agents/multi_agent/statistic_agent.py
git commit -m "feat: wire PlayerStatsTool into StatisticAgent"
```

---

## Task 7: Initialize Production DB

- [ ] **Step 1: Run init script**

```bash
DATABASE_URL=<railway-url> python scripts/pipelines/init_player_stats.py
```

Expected:
```
INFO - Initializing player stats for seasons: ['2023-24', '2024-25', '2025-26']
INFO - Season 2023-24:
INFO -   Upserted — pg:540 tot:540 p36:540
INFO -   Value metrics calculated
INFO - Season 2024-25:
INFO -   Upserted — pg:569 tot:569 p36:569
INFO -   Value metrics calculated
INFO - Season 2025-26:
INFO -   Upserted — pg:570 tot:570 p36:570
INFO -   Value metrics calculated
```

- [ ] **Step 2: Verify spot check**

```bash
psql <railway-url> -c "
  SELECT player_name, gp, pts, rv, pv, three_v, z_pts, z_fg
  FROM bball_monsters_player_stats_pg
  WHERE season='2025-26'
  ORDER BY rv DESC LIMIT 10;"
```

- [ ] **Step 3: Push to origin**

```bash
git push origin main
```

---

## Task 8: Daily Railway Cron

**Manual setup — Railway dashboard only.**

- [ ] **Step 1: Create `player-stats-cron` service**
  1. Railway → `clutchai` project → production
  2. New Service → Empty → name `player-stats-cron`
  3. Source → GitHub repo, branch `main`
  4. Start Command: `python scripts/pipelines/update_player_stats.py`
  5. Cron Schedule: `0 8 * * *`
  6. Variables → `DATABASE_URL` (copy from Postgres service)

- [ ] **Step 2: Disable in staging**
  Staging → `player-stats-cron` → Cron Schedule: `0 0 1 1 *`

- [ ] **Step 3: Verify logs after manual trigger**

---

## Self-Review

**Spec coverage:**
- ✅ 3 tables with correct names: `bball_monsters_player_stats_pg/total/p36`
- ✅ Per-game, totals, per-36 stat columns (no prefix, table is the context)
- ✅ 9 z-scores per table: `z_pts, z_reb, z_ast, z_stl, z_blk, z_3ptm, z_tov, z_fg, z_ft`
- ✅ FG%/FT% volume-weighted z-scores
- ✅ `rv` = sum of 9 z-scores; `three_v` = `z_3ptm`; `pv` = Yahoo points vs 150th player
- ✅ `pv` omitted from total table
- ✅ Pool = top 150 by `min` from pg table (shared anchor across all 3 tables)
- ✅ Last 3 seasons init; daily cron skips July–September
- ✅ `basketball_monster_stats` tool accepts `stat_type` (pg/total/p36)

**Type consistency:**
- `mgr.fetch(season)` → `(pg_df, tot_df, p36_df)` → `mgr.upsert_all(pg, tot, p36, season)` ✅
- `calc.calculate(season, table='all')` reads pg table first for pool anchor in all contexts ✅
- `_TABLES` dict in tool matches table names in manager ✅
- `z_3ptm` column name consistent across DDL, calculator, and tool description ✅
