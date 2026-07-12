# Player Stats Database Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Postgres-backed player stats layer — game logs, team schedules, opponent defense rankings — with a nightly cron pipeline and 4 StatsAgent tools that replace live NBA API calls for in-season analysis.

**Architecture:** Three new data managers in `data/postgres/` persist stats from the NBA API; `PlayerStatsManager` gains std dev columns and an `update_std_devs()` method. A new `update_player_stats_db.py` pipeline cron orchestrates nightly refresh. Four LangChain tools in `agents/tools/player_stats_db.py` give `StatisticAgent` fast, DB-backed query capabilities.

**Tech Stack:** PostgreSQL (Railway), SQLAlchemy text queries, nba_api, LangChain `@tool`, pytest + unittest.mock

## Global Constraints

- Branch: `feature/player_db` (created from `main` at the start of Task 1)
- All tests go in `tests/` flat directory — no subdirectories
- All NBA API calls in managers must have `time.sleep(1)` between requests
- DB tool classes follow `PlayerStatsTool` pattern: `connection or PostgresConnection()` fallback
- All test manager factories follow the `_make_manager()` pattern in `tests/test_player_stats_manager.py`
- `is_nba_season()`, `current_season()`, `_PG_TABLE`, `_TOT_TABLE`, `_P36_TABLE` all live in `data/postgres/player_stats.py` — import, never redefine
- `_STD_DEV_COLS` defined in `data/postgres/player_stats.py` — import it in tests

---

## File Map

**Create:**
- `data/postgres/player_game_logs.py` — `PlayerGameLogsManager`: create table, upsert daily game logs, compute std dev
- `data/postgres/team_schedules.py` — `TeamScheduleManager`: create table, load full season schedule, patch postponements
- `data/postgres/opponent_defense.py` — `OpponentDefenseManager`: create table, fetch + upsert weekly opponent rankings
- `scripts/__init__.py` — empty, makes scripts a package for testing
- `scripts/pipelines/__init__.py` — empty, makes scripts.pipelines a package for testing
- `scripts/pipelines/update_player_stats_db.py` — pipeline entry point; orchestrates all daily + weekly steps
- `agents/tools/player_stats_db.py` — `PlayerStatsDbTool`: 4 LangChain tools for `StatisticAgent`
- `tests/test_player_stats_std_dev.py`
- `tests/test_player_game_logs.py`
- `tests/test_team_schedules.py`
- `tests/test_opponent_defense.py`
- `tests/test_update_player_stats_db.py`
- `tests/test_player_stats_db_tools.py`

**Modify:**
- `data/postgres/player_stats.py` — add `_STD_DEV_COLS`, std dev DDL to `_make_create_sql`, `migrate_std_dev_cols()` function, `update_std_devs()` method on `PlayerStatsManager`
- `agents/multi_agent/statistic_agent.py` — add `PlayerStatsDbTool`, extend system prompt

---

## Task 1: Std Dev Schema — Add Columns to Existing Aggregate Tables

**Files:**
- Modify: `data/postgres/player_stats.py`
- Test: `tests/test_player_stats_std_dev.py`

**Interfaces:**
- Produces: `_STD_DEV_COLS: list[str]`, `migrate_std_dev_cols(connection: PostgresConnection) -> bool`, `PlayerStatsManager.update_std_devs(std_devs: dict, season: str) -> int`
- `std_devs` shape: `{player_id: int → {std_dev_pts, std_dev_reb, std_dev_ast, std_dev_stl, std_dev_blk, std_dev_to, std_dev_fgp, std_dev_3pp, std_dev_ftp: float}}`

- [ ] **Step 1: Create branch**

```bash
git checkout main && git pull origin main
git checkout -b feature/player_db
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_player_stats_std_dev.py`:

```python
from unittest.mock import MagicMock, patch
from data.postgres.player_stats import PlayerStatsManager, migrate_std_dev_cols, _STD_DEV_COLS


def _make_manager():
    mock_conn = MagicMock()
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_pg = MagicMock()
    mock_pg.get_engine.return_value = mock_engine
    return PlayerStatsManager(mock_pg), mock_conn, mock_pg


def test_std_dev_cols_has_nine_entries():
    assert len(_STD_DEV_COLS) == 9

def test_migrate_std_dev_cols_returns_true():
    _, _, mock_pg = _make_manager()
    assert migrate_std_dev_cols(mock_pg) is True

def test_migrate_std_dev_cols_issues_alter_for_pg_table():
    _, mock_conn, mock_pg = _make_manager()
    migrate_std_dev_cols(mock_pg)
    sqls = [call[0][0].text for call in mock_conn.execute.call_args_list]
    assert any('bball_monsters_player_stats_pg' in s for s in sqls)

def test_migrate_std_dev_cols_issues_alter_for_total_table():
    _, mock_conn, mock_pg = _make_manager()
    migrate_std_dev_cols(mock_pg)
    sqls = [call[0][0].text for call in mock_conn.execute.call_args_list]
    assert any('bball_monsters_player_stats_total' in s for s in sqls)

def test_migrate_std_dev_cols_issues_alter_for_p36_table():
    _, mock_conn, mock_pg = _make_manager()
    migrate_std_dev_cols(mock_pg)
    sqls = [call[0][0].text for call in mock_conn.execute.call_args_list]
    assert any('bball_monsters_player_stats_p36' in s for s in sqls)

def test_migrate_std_dev_cols_includes_all_std_dev_cols():
    _, mock_conn, mock_pg = _make_manager()
    migrate_std_dev_cols(mock_pg)
    combined = ' '.join(call[0][0].text for call in mock_conn.execute.call_args_list)
    for col in _STD_DEV_COLS:
        assert col in combined

def test_migrate_std_dev_cols_returns_false_on_error():
    mock_pg = MagicMock()
    mock_pg.get_engine.side_effect = Exception("DB down")
    assert migrate_std_dev_cols(mock_pg) is False

def test_update_std_devs_returns_zero_for_empty():
    mgr, _, _ = _make_manager()
    assert mgr.update_std_devs({}, '2025-26') == 0

def test_update_std_devs_returns_player_count():
    mgr, _, _ = _make_manager()
    devs = {
        2544:   {c: 1.5 for c in _STD_DEV_COLS},
        203999: {c: 2.0 for c in _STD_DEV_COLS},
    }
    assert mgr.update_std_devs(devs, '2025-26') == 2

def test_update_std_devs_executes_update_for_pg_table():
    mgr, mock_conn, _ = _make_manager()
    mgr.update_std_devs({2544: {c: 1.5 for c in _STD_DEV_COLS}}, '2025-26')
    sqls = [call[0][0].text for call in mock_conn.execute.call_args_list]
    assert any('bball_monsters_player_stats_pg' in s for s in sqls)

def test_update_std_devs_executes_update_for_total_table():
    mgr, mock_conn, _ = _make_manager()
    mgr.update_std_devs({2544: {c: 1.5 for c in _STD_DEV_COLS}}, '2025-26')
    sqls = [call[0][0].text for call in mock_conn.execute.call_args_list]
    assert any('bball_monsters_player_stats_total' in s for s in sqls)

def test_update_std_devs_executes_update_for_p36_table():
    mgr, mock_conn, _ = _make_manager()
    mgr.update_std_devs({2544: {c: 1.5 for c in _STD_DEV_COLS}}, '2025-26')
    sqls = [call[0][0].text for call in mock_conn.execute.call_args_list]
    assert any('bball_monsters_player_stats_p36' in s for s in sqls)

def test_update_std_devs_sql_uses_update_statement():
    mgr, mock_conn, _ = _make_manager()
    mgr.update_std_devs({2544: {c: 1.5 for c in _STD_DEV_COLS}}, '2025-26')
    sqls = [call[0][0].text for call in mock_conn.execute.call_args_list]
    assert all('UPDATE' in s for s in sqls)

def test_update_std_devs_commits():
    mgr, mock_conn, _ = _make_manager()
    mgr.update_std_devs({2544: {c: 1.5 for c in _STD_DEV_COLS}}, '2025-26')
    mock_conn.commit.assert_called_once()

def test_create_tables_sql_includes_std_dev_cols():
    mgr, mock_conn, _ = _make_manager()
    mgr.create_tables()
    combined = ' '.join(call[0][0].text for call in mock_conn.execute.call_args_list)
    for col in _STD_DEV_COLS:
        assert col in combined
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_player_stats_std_dev.py -v
```
Expected: ImportError or AttributeError — `_STD_DEV_COLS`, `migrate_std_dev_cols`, `update_std_devs` not yet defined.

- [ ] **Step 4: Implement in `data/postgres/player_stats.py`**

After the `_Z_COLS` / `_PG_VALUE_COLS` block (around line 24), add:

```python
_STD_DEV_COLS = [
    'std_dev_pts', 'std_dev_reb', 'std_dev_ast', 'std_dev_stl', 'std_dev_blk',
    'std_dev_to', 'std_dev_fgp', 'std_dev_3pp', 'std_dev_ftp',
]
```

Update `_make_create_sql` to include std dev DDL (add after `value_ddl` line):

```python
def _make_create_sql(table: str, stat_cols: list, value_cols: list) -> text:
    stat_ddl     = ',\n        '.join(f'{_stat_sql_col(c)} FLOAT' for c in stat_cols)
    z_ddl        = ',\n        '.join(f'"{c}" FLOAT' for c in _Z_COLS)
    value_ddl    = ',\n        '.join(f'{c} FLOAT' for c in value_cols)
    std_dev_ddl  = ',\n        '.join(f'{c} FLOAT' for c in _STD_DEV_COLS)
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
            {std_dev_ddl},
            updated_at        TIMESTAMP DEFAULT NOW(),
            PRIMARY KEY (player_id, season)
        )
    """)
```

After the `_UPSERT_P36` line, add module-level migration function:

```python
def migrate_std_dev_cols(connection: PostgresConnection) -> bool:
    """Add std dev columns to existing aggregate tables. Safe to run multiple times."""
    adds = ', '.join(f'ADD COLUMN IF NOT EXISTS {c} FLOAT' for c in _STD_DEV_COLS)
    try:
        with connection.get_engine().connect() as conn:
            for table in [_PG_TABLE, _TOT_TABLE, _P36_TABLE]:
                conn.execute(text(f'ALTER TABLE {table} {adds}'))
            conn.commit()
        return True
    except Exception as e:
        logger.error(f"Failed to migrate std dev columns: {e}")
        return False
```

Add `update_std_devs` method to `PlayerStatsManager` after `upsert_all`:

```python
def update_std_devs(self, std_devs: dict, season: str) -> int:
    """Write std dev values back to all 3 aggregate tables.

    Args:
        std_devs: {player_id: {std_dev_pts, std_dev_reb, std_dev_ast, std_dev_stl,
                               std_dev_blk, std_dev_to, std_dev_fgp, std_dev_3pp,
                               std_dev_ftp: float}}
    Returns: Number of players updated.
    """
    if not std_devs:
        return 0
    set_clause = ', '.join(f'{c} = :{c}' for c in _STD_DEV_COLS)
    sqls = {
        t: text(f'UPDATE {t} SET {set_clause}, updated_at = NOW() WHERE player_id = :player_id AND season = :season')
        for t in [_PG_TABLE, _TOT_TABLE, _P36_TABLE]
    }
    updated = 0
    with self.connection.get_engine().connect() as conn:
        for player_id, devs in std_devs.items():
            params = {'player_id': player_id, 'season': season, **devs}
            for sql in sqls.values():
                conn.execute(sql, params)
            updated += 1
        conn.commit()
    return updated
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_player_stats_std_dev.py -v
```
Expected: All 14 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add data/postgres/player_stats.py tests/test_player_stats_std_dev.py
git commit -m "feat: add std dev columns and migrate/update methods to PlayerStatsManager"
```

---

## Task 2: `player_game_logs` Table + Manager

**Files:**
- Create: `data/postgres/player_game_logs.py`
- Test: `tests/test_player_game_logs.py`

**Interfaces:**
- Consumes: `PostgresConnection` from `data.postgres.connection`
- Produces: `PlayerGameLogsManager`, `_parse_min(min_str) -> Optional[float]`
- `fetch_and_upsert_date(game_date: date, season: str) -> int`
- `get_game_ids_on_date(game_date: date) -> list[str]`
- `compute_std_dev(season: str) -> dict` — returns `{player_id: {std_dev_pts: x, ...}}` with keys matching `_STD_DEV_COLS`

- [ ] **Step 1: Write failing tests**

Create `tests/test_player_game_logs.py`:

```python
import math
import pandas as pd
from datetime import date
from unittest.mock import MagicMock, patch

from data.postgres.player_game_logs import PlayerGameLogsManager, _parse_min


# --- _parse_min ---

def test_parse_min_mm_ss_string():
    assert abs(_parse_min('35:23') - (35 + 23/60)) < 0.001

def test_parse_min_float_passthrough():
    assert _parse_min(35.5) == 35.5

def test_parse_min_int_passthrough():
    assert _parse_min(32) == 32.0

def test_parse_min_none_returns_none():
    assert _parse_min(None) is None

def test_parse_min_nan_returns_none():
    assert _parse_min(float('nan')) is None


# --- helpers ---

def _make_manager():
    mock_conn = MagicMock()
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_pg = MagicMock()
    mock_pg.get_engine.return_value = mock_engine
    return PlayerGameLogsManager(mock_pg), mock_conn


_BOX_ROW = {
    'PLAYER_ID': 2544, 'TEAM_ABBREVIATION': 'LAL',
    'MIN': '35:00', 'PTS': 28.0, 'REB': 7.0, 'AST': 9.0,
    'STL': 1.0, 'BLK': 0.0, 'TO': 3.0,
    'FGM': 10.0, 'FGA': 20.0, 'FG_PCT': 0.500,
    'FG3M': 2.0, 'FG3A': 6.0, 'FG3_PCT': 0.333,
    'FTM': 6.0, 'FTA': 8.0, 'FT_PCT': 0.750, 'PLUS_MINUS': 5.0,
}


# --- create_table ---

def test_create_table_returns_true():
    mgr, _ = _make_manager()
    assert mgr.create_table() is True

def test_create_table_sql_has_player_game_logs():
    mgr, mock_conn = _make_manager()
    mgr.create_table()
    sql = mock_conn.execute.call_args[0][0].text
    assert 'player_game_logs' in sql

def test_create_table_returns_false_on_error():
    mock_pg = MagicMock()
    mock_pg.get_engine.side_effect = Exception("DB down")
    assert PlayerGameLogsManager(mock_pg).create_table() is False


# --- _get_game_ids ---

def test_get_game_ids_returns_list():
    mgr, _ = _make_manager()
    mock_sb = MagicMock()
    mock_sb.get_data_frames.return_value = [pd.DataFrame({'GAME_ID': ['0022500001', '0022500002']})]
    with patch('data.postgres.player_game_logs.scoreboardv2.ScoreboardV2', return_value=mock_sb):
        ids = mgr._get_game_ids(date(2026, 3, 15))
    assert ids == ['0022500001', '0022500002']

def test_get_game_ids_returns_empty_list_when_no_games():
    mgr, _ = _make_manager()
    mock_sb = MagicMock()
    mock_sb.get_data_frames.return_value = [pd.DataFrame({'GAME_ID': []})]
    with patch('data.postgres.player_game_logs.scoreboardv2.ScoreboardV2', return_value=mock_sb):
        ids = mgr._get_game_ids(date(2026, 7, 4))
    assert ids == []


# --- _get_player_rows ---

def test_get_player_rows_returns_list_of_dicts():
    mgr, _ = _make_manager()
    mock_bs = MagicMock()
    mock_bs.get_data_frames.return_value = [pd.DataFrame([_BOX_ROW])]
    with patch('data.postgres.player_game_logs.boxscoretraditionalv2.BoxScoreTraditionalV2', return_value=mock_bs):
        rows = mgr._get_player_rows('0022500001', date(2026, 3, 15), '2025-26')
    assert len(rows) == 1
    assert rows[0]['player_id'] == 2544
    assert rows[0]['pts'] == 28.0
    assert rows[0]['to'] == 3.0

def test_get_player_rows_skips_null_player_id():
    mgr, _ = _make_manager()
    null_row = {**_BOX_ROW, 'PLAYER_ID': float('nan')}
    mock_bs = MagicMock()
    mock_bs.get_data_frames.return_value = [pd.DataFrame([null_row])]
    with patch('data.postgres.player_game_logs.boxscoretraditionalv2.BoxScoreTraditionalV2', return_value=mock_bs):
        rows = mgr._get_player_rows('0022500001', date(2026, 3, 15), '2025-26')
    assert rows == []

def test_get_player_rows_parses_min_string():
    mgr, _ = _make_manager()
    mock_bs = MagicMock()
    mock_bs.get_data_frames.return_value = [pd.DataFrame([_BOX_ROW])]
    with patch('data.postgres.player_game_logs.boxscoretraditionalv2.BoxScoreTraditionalV2', return_value=mock_bs):
        rows = mgr._get_player_rows('0022500001', date(2026, 3, 15), '2025-26')
    assert rows[0]['min'] == 35.0


# --- _upsert_rows ---

def test_upsert_rows_returns_zero_for_empty():
    mgr, _ = _make_manager()
    assert mgr._upsert_rows([]) == 0

def test_upsert_rows_returns_row_count():
    mgr, _ = _make_manager()
    row = {
        'player_id': 2544, 'game_id': '001', 'game_date': date(2026, 3, 15),
        'season': '2025-26', 'team_abbreviation': 'LAL', 'min': 35.0,
        'pts': 28.0, 'reb': 7.0, 'ast': 9.0, 'stl': 1.0, 'blk': 0.0, 'to': 3.0,
        'fgm': 10.0, 'fga': 20.0, 'fg_pct': 0.5, 'fg3m': 2.0, 'fg3a': 6.0,
        'fg3_pct': 0.333, 'ftm': 6.0, 'fta': 8.0, 'ft_pct': 0.75, 'plus_minus': 5.0,
    }
    assert mgr._upsert_rows([row]) == 1

def test_upsert_rows_sql_has_on_conflict():
    mgr, mock_conn = _make_manager()
    row = {
        'player_id': 2544, 'game_id': '001', 'game_date': date(2026, 3, 15),
        'season': '2025-26', 'team_abbreviation': 'LAL', 'min': 35.0,
        'pts': 28.0, 'reb': 7.0, 'ast': 9.0, 'stl': 1.0, 'blk': 0.0, 'to': 3.0,
        'fgm': 10.0, 'fga': 20.0, 'fg_pct': 0.5, 'fg3m': 2.0, 'fg3a': 6.0,
        'fg3_pct': 0.333, 'ftm': 6.0, 'fta': 8.0, 'ft_pct': 0.75, 'plus_minus': 5.0,
    }
    mgr._upsert_rows([row])
    sql = mock_conn.execute.call_args[0][0].text
    assert 'ON CONFLICT' in sql


# --- fetch_and_upsert_date ---

def test_fetch_and_upsert_date_returns_zero_when_no_games():
    mgr, _ = _make_manager()
    mock_sb = MagicMock()
    mock_sb.get_data_frames.return_value = [pd.DataFrame({'GAME_ID': []})]
    with patch('data.postgres.player_game_logs.scoreboardv2.ScoreboardV2', return_value=mock_sb):
        with patch('data.postgres.player_game_logs.time.sleep'):
            result = mgr.fetch_and_upsert_date(date(2026, 7, 4), '2025-26')
    assert result == 0

def test_fetch_and_upsert_date_sleeps_once_per_game():
    mgr, _ = _make_manager()
    mock_sb = MagicMock()
    mock_sb.get_data_frames.return_value = [pd.DataFrame({'GAME_ID': ['001', '002']})]
    mock_bs = MagicMock()
    mock_bs.get_data_frames.return_value = [pd.DataFrame(columns=list(_BOX_ROW.keys()))]
    with patch('data.postgres.player_game_logs.scoreboardv2.ScoreboardV2', return_value=mock_sb):
        with patch('data.postgres.player_game_logs.boxscoretraditionalv2.BoxScoreTraditionalV2', return_value=mock_bs):
            with patch('data.postgres.player_game_logs.time.sleep') as mock_sleep:
                mgr.fetch_and_upsert_date(date(2026, 3, 15), '2025-26')
    assert mock_sleep.call_count == 2


# --- compute_std_dev ---

def test_compute_std_dev_queries_player_game_logs():
    mgr, mock_conn = _make_manager()
    mock_conn.execute.return_value = iter([])
    mgr.compute_std_dev('2025-26')
    sql = mock_conn.execute.call_args[0][0].text
    assert 'player_game_logs' in sql
    assert ':season' in sql

def test_compute_std_dev_returns_player_dict():
    mgr, mock_conn = _make_manager()
    mock_conn.execute.return_value = iter([
        (2544, 3.5, 2.1, 1.8, 0.5, 0.3, 0.9, 0.05, 0.08, 0.06)
    ])
    result = mgr.compute_std_dev('2025-26')
    assert 2544 in result
    assert result[2544]['std_dev_pts'] == 3.5
    assert result[2544]['std_dev_fgp'] == 0.05

def test_compute_std_dev_returns_empty_dict_when_no_data():
    mgr, mock_conn = _make_manager()
    mock_conn.execute.return_value = iter([])
    assert mgr.compute_std_dev('2025-26') == {}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_player_game_logs.py -v
```
Expected: ImportError — module does not exist yet.

- [ ] **Step 3: Implement `data/postgres/player_game_logs.py`**

```python
import time
from datetime import date
from typing import Optional

import pandas as pd
from sqlalchemy import text

from nba_api.stats.endpoints import boxscoretraditionalv2, scoreboardv2
from data.postgres.connection import PostgresConnection
from logger import get_logger

logger = get_logger(__name__)

_CREATE_SQL = text("""
    CREATE TABLE IF NOT EXISTS player_game_logs (
        player_id         INTEGER      NOT NULL,
        game_id           VARCHAR(20)  NOT NULL,
        game_date         DATE         NOT NULL,
        season            VARCHAR(10)  NOT NULL,
        team_abbreviation VARCHAR(10),
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
    )
""")

_UPSERT_SQL = text("""
    INSERT INTO player_game_logs (
        player_id, game_id, game_date, season, team_abbreviation,
        min, pts, reb, ast, stl, blk, "to",
        fgm, fga, fg_pct, fg3m, fg3a, fg3_pct, ftm, fta, ft_pct,
        plus_minus, updated_at
    ) VALUES (
        :player_id, :game_id, :game_date, :season, :team_abbreviation,
        :min, :pts, :reb, :ast, :stl, :blk, :to,
        :fgm, :fga, :fg_pct, :fg3m, :fg3a, :fg3_pct, :ftm, :fta, :ft_pct,
        :plus_minus, NOW()
    )
    ON CONFLICT (player_id, game_id) DO UPDATE SET
        pts = EXCLUDED.pts, reb = EXCLUDED.reb, ast = EXCLUDED.ast,
        stl = EXCLUDED.stl, blk = EXCLUDED.blk, "to" = EXCLUDED."to",
        fgm = EXCLUDED.fgm, fga = EXCLUDED.fga, fg_pct = EXCLUDED.fg_pct,
        fg3m = EXCLUDED.fg3m, fg3a = EXCLUDED.fg3a, fg3_pct = EXCLUDED.fg3_pct,
        ftm = EXCLUDED.ftm, fta = EXCLUDED.fta, ft_pct = EXCLUDED.ft_pct,
        plus_minus = EXCLUDED.plus_minus, min = EXCLUDED.min,
        updated_at = NOW()
""")


def _parse_min(min_str) -> Optional[float]:
    """Convert 'MM:SS' box score string to float minutes."""
    if min_str is None:
        return None
    if isinstance(min_str, float) and pd.isna(min_str):
        return None
    if isinstance(min_str, (int, float)):
        return float(min_str)
    parts = str(min_str).split(':')
    return float(parts[0]) + float(parts[1]) / 60 if len(parts) == 2 else float(parts[0])


class PlayerGameLogsManager:

    def __init__(self, connection: Optional[PostgresConnection] = None):
        self.connection = connection or PostgresConnection()

    def create_table(self) -> bool:
        try:
            with self.connection.get_engine().connect() as conn:
                conn.execute(_CREATE_SQL)
                conn.commit()
            logger.info("Created/verified player_game_logs table")
            return True
        except Exception as e:
            logger.error(f"Failed to create player_game_logs table: {e}")
            return False

    def _get_game_ids(self, game_date: date) -> list:
        date_str = game_date.strftime('%Y-%m-%d')
        sb = scoreboardv2.ScoreboardV2(game_date=date_str, timeout=30)
        df = sb.get_data_frames()[0]  # GameHeader
        return df['GAME_ID'].tolist()

    def _get_player_rows(self, game_id: str, game_date: date, season: str) -> list:
        bs = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id, timeout=30)
        df = bs.get_data_frames()[0]  # PlayerStats
        rows = []
        for _, row in df.iterrows():
            if pd.isna(row.get('PLAYER_ID')):
                continue
            rows.append({
                'player_id':         int(row['PLAYER_ID']),
                'game_id':           game_id,
                'game_date':         game_date,
                'season':            season,
                'team_abbreviation': row.get('TEAM_ABBREVIATION'),
                'min':               _parse_min(row.get('MIN')),
                'pts':               row.get('PTS'),
                'reb':               row.get('REB'),
                'ast':               row.get('AST'),
                'stl':               row.get('STL'),
                'blk':               row.get('BLK'),
                'to':                row.get('TO'),
                'fgm':               row.get('FGM'),
                'fga':               row.get('FGA'),
                'fg_pct':            row.get('FG_PCT'),
                'fg3m':              row.get('FG3M'),
                'fg3a':              row.get('FG3A'),
                'fg3_pct':           row.get('FG3_PCT'),
                'ftm':               row.get('FTM'),
                'fta':               row.get('FTA'),
                'ft_pct':            row.get('FT_PCT'),
                'plus_minus':        row.get('PLUS_MINUS'),
            })
        return rows

    def _upsert_rows(self, rows: list) -> int:
        if not rows:
            return 0
        with self.connection.get_engine().connect() as conn:
            for row in rows:
                conn.execute(_UPSERT_SQL, row)
            conn.commit()
        return len(rows)

    def fetch_and_upsert_date(self, game_date: date, season: str) -> int:
        game_ids = self._get_game_ids(game_date)
        if not game_ids:
            logger.info(f"No games on {game_date}")
            return 0
        total = 0
        for game_id in game_ids:
            time.sleep(1)
            rows = self._get_player_rows(game_id, game_date, season)
            total += self._upsert_rows(rows)
        logger.info(f"Upserted {total} player game log rows for {game_date}")
        return total

    def get_game_ids_on_date(self, game_date: date) -> list:
        sql = text('SELECT DISTINCT game_id FROM player_game_logs WHERE game_date = :d')
        with self.connection.get_engine().connect() as conn:
            result = conn.execute(sql, {'d': game_date})
            return [row[0] for row in result]

    def compute_std_dev(self, season: str) -> dict:
        """Returns {player_id: {std_dev_pts, std_dev_reb, ..., std_dev_ftp}} for all players."""
        sql = text("""
            SELECT
                player_id,
                STDDEV(pts)     AS std_dev_pts,
                STDDEV(reb)     AS std_dev_reb,
                STDDEV(ast)     AS std_dev_ast,
                STDDEV(stl)     AS std_dev_stl,
                STDDEV(blk)     AS std_dev_blk,
                STDDEV("to")    AS std_dev_to,
                STDDEV(fg_pct)  AS std_dev_fgp,
                STDDEV(fg3_pct) AS std_dev_3pp,
                STDDEV(ft_pct)  AS std_dev_ftp
            FROM player_game_logs
            WHERE season = :season
            GROUP BY player_id
        """)
        with self.connection.get_engine().connect() as conn:
            result = conn.execute(sql, {'season': season})
            return {
                row[0]: {
                    'std_dev_pts': row[1], 'std_dev_reb': row[2], 'std_dev_ast': row[3],
                    'std_dev_stl': row[4], 'std_dev_blk': row[5], 'std_dev_to':  row[6],
                    'std_dev_fgp': row[7], 'std_dev_3pp': row[8], 'std_dev_ftp': row[9],
                }
                for row in result
            }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_player_game_logs.py -v
```
Expected: All 19 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add data/postgres/player_game_logs.py tests/test_player_game_logs.py
git commit -m "feat: add PlayerGameLogsManager with daily upsert and std dev computation"
```

---

## Task 3: `team_schedules` Table + Manager

**Files:**
- Create: `data/postgres/team_schedules.py`
- Test: `tests/test_team_schedules.py`

**Interfaces:**
- Produces: `TeamScheduleManager`, `_parse_matchup(matchup: str, team_abbr: str) -> tuple[str, str]`
- `create_table() -> bool`, `is_loaded(season: str) -> bool`, `load_season(season: str) -> int`
- `ensure_loaded(season: str) -> None` — calls `load_season` only if not yet loaded
- `patch_postponements(game_date: date, completed_game_ids: list[str]) -> int`

- [ ] **Step 1: Write failing tests**

Create `tests/test_team_schedules.py`:

```python
import pandas as pd
from datetime import date
from unittest.mock import MagicMock, patch

from data.postgres.team_schedules import TeamScheduleManager, _parse_matchup


# --- _parse_matchup ---

def test_parse_matchup_home_game():
    home_away, opp = _parse_matchup('BOS vs. MIA', 'BOS')
    assert home_away == 'home'
    assert opp == 'MIA'

def test_parse_matchup_away_game():
    home_away, opp = _parse_matchup('BOS @ MIA', 'BOS')
    assert home_away == 'away'
    assert opp == 'MIA'


# --- helpers ---

def _make_manager():
    mock_conn = MagicMock()
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_pg = MagicMock()
    mock_pg.get_engine.return_value = mock_engine
    return TeamScheduleManager(mock_pg), mock_conn


_SCHEDULE_ROW = {
    'TEAM_ID': 1610612738, 'TEAM_ABBREVIATION': 'BOS',
    'GAME_ID': '0022500001', 'GAME_DATE': '2025-10-22',
    'MATCHUP': 'BOS vs. MIA', 'WL': 'W',
}


# --- create_table ---

def test_create_table_returns_true():
    mgr, _ = _make_manager()
    assert mgr.create_table() is True

def test_create_table_sql_has_team_schedules():
    mgr, mock_conn = _make_manager()
    mgr.create_table()
    sql = mock_conn.execute.call_args[0][0].text
    assert 'team_schedules' in sql

def test_create_table_returns_false_on_error():
    mock_pg = MagicMock()
    mock_pg.get_engine.side_effect = Exception("DB down")
    assert TeamScheduleManager(mock_pg).create_table() is False


# --- is_loaded ---

def test_is_loaded_true_when_rows_exist():
    mgr, mock_conn = _make_manager()
    mock_conn.execute.return_value.scalar.return_value = 100
    assert mgr.is_loaded('2025-26') is True

def test_is_loaded_false_when_no_rows():
    mgr, mock_conn = _make_manager()
    mock_conn.execute.return_value.scalar.return_value = 0
    assert mgr.is_loaded('2025-26') is False


# --- load_season ---

def test_load_season_calls_league_game_finder():
    mgr, _ = _make_manager()
    mock_lgf = MagicMock()
    mock_lgf.get_data_frames.return_value = [pd.DataFrame([_SCHEDULE_ROW])]
    with patch('data.postgres.team_schedules.leaguegamefinder.LeagueGameFinder', return_value=mock_lgf) as mock_cls:
        mgr.load_season('2025-26')
    mock_cls.assert_called_once()

def test_load_season_returns_row_count():
    mgr, _ = _make_manager()
    mock_lgf = MagicMock()
    mock_lgf.get_data_frames.return_value = [pd.DataFrame([_SCHEDULE_ROW, _SCHEDULE_ROW])]
    with patch('data.postgres.team_schedules.leaguegamefinder.LeagueGameFinder', return_value=mock_lgf):
        assert mgr.load_season('2025-26') == 2

def test_load_season_sql_has_on_conflict():
    mgr, mock_conn = _make_manager()
    mock_lgf = MagicMock()
    mock_lgf.get_data_frames.return_value = [pd.DataFrame([_SCHEDULE_ROW])]
    with patch('data.postgres.team_schedules.leaguegamefinder.LeagueGameFinder', return_value=mock_lgf):
        mgr.load_season('2025-26')
    sql = mock_conn.execute.call_args[0][0].text
    assert 'ON CONFLICT' in sql


# --- ensure_loaded ---

def test_ensure_loaded_calls_load_season_when_empty():
    mgr, mock_conn = _make_manager()
    mock_conn.execute.return_value.scalar.return_value = 0
    mock_lgf = MagicMock()
    mock_lgf.get_data_frames.return_value = [pd.DataFrame(columns=list(_SCHEDULE_ROW.keys()))]
    with patch('data.postgres.team_schedules.leaguegamefinder.LeagueGameFinder', return_value=mock_lgf) as mock_cls:
        mgr.ensure_loaded('2025-26')
    mock_cls.assert_called_once()

def test_ensure_loaded_skips_load_when_already_loaded():
    mgr, mock_conn = _make_manager()
    mock_conn.execute.return_value.scalar.return_value = 1000
    with patch('data.postgres.team_schedules.leaguegamefinder.LeagueGameFinder') as mock_cls:
        mgr.ensure_loaded('2025-26')
    mock_cls.assert_not_called()


# --- patch_postponements ---

def test_patch_postponements_marks_missing_game():
    mgr, mock_conn = _make_manager()
    mock_conn.execute.return_value = iter([('0022500001',), ('0022500002',)])
    updated = mgr.patch_postponements(date(2026, 3, 15), completed_game_ids=['0022500001'])
    assert updated == 1

def test_patch_postponements_returns_zero_when_all_completed():
    mgr, mock_conn = _make_manager()
    mock_conn.execute.return_value = iter([('0022500001',)])
    updated = mgr.patch_postponements(date(2026, 3, 15), completed_game_ids=['0022500001'])
    assert updated == 0

def test_patch_postponements_returns_zero_when_no_scheduled_games():
    mgr, mock_conn = _make_manager()
    mock_conn.execute.return_value = iter([])
    assert mgr.patch_postponements(date(2026, 7, 4), completed_game_ids=[]) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_team_schedules.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `data/postgres/team_schedules.py`**

```python
from datetime import date
from typing import Optional

import pandas as pd
from sqlalchemy import text

from nba_api.stats.endpoints import leaguegamefinder
from data.postgres.connection import PostgresConnection
from logger import get_logger

logger = get_logger(__name__)

_CREATE_SQL = text("""
    CREATE TABLE IF NOT EXISTS team_schedules (
        team_id           INTEGER      NOT NULL,
        team_abbreviation VARCHAR(10)  NOT NULL,
        game_id           VARCHAR(20)  NOT NULL,
        game_date         DATE         NOT NULL,
        season            VARCHAR(10)  NOT NULL,
        home_away         VARCHAR(10),
        opponent_abbr     VARCHAR(10),
        postponed         BOOLEAN      DEFAULT FALSE,
        updated_at        TIMESTAMP DEFAULT NOW(),
        PRIMARY KEY (team_id, game_id)
    )
""")

_UPSERT_SQL = text("""
    INSERT INTO team_schedules (
        team_id, team_abbreviation, game_id, game_date, season,
        home_away, opponent_abbr, postponed, updated_at
    ) VALUES (
        :team_id, :team_abbreviation, :game_id, :game_date, :season,
        :home_away, :opponent_abbr, FALSE, NOW()
    )
    ON CONFLICT (team_id, game_id) DO UPDATE SET
        game_date = EXCLUDED.game_date,
        home_away = EXCLUDED.home_away,
        opponent_abbr = EXCLUDED.opponent_abbr,
        updated_at = NOW()
""")


def _parse_matchup(matchup: str, team_abbr: str) -> tuple:
    """Returns (home_away, opponent_abbr) from MATCHUP string like 'BOS vs. MIA' or 'BOS @ MIA'."""
    parts = matchup.split()
    opponent_abbr = parts[-1]
    home_away = 'home' if 'vs.' in matchup else 'away'
    return home_away, opponent_abbr


class TeamScheduleManager:

    def __init__(self, connection: Optional[PostgresConnection] = None):
        self.connection = connection or PostgresConnection()

    def create_table(self) -> bool:
        try:
            with self.connection.get_engine().connect() as conn:
                conn.execute(_CREATE_SQL)
                conn.commit()
            logger.info("Created/verified team_schedules table")
            return True
        except Exception as e:
            logger.error(f"Failed to create team_schedules table: {e}")
            return False

    def is_loaded(self, season: str) -> bool:
        sql = text('SELECT COUNT(*) FROM team_schedules WHERE season = :season')
        with self.connection.get_engine().connect() as conn:
            return conn.execute(sql, {'season': season}).scalar() > 0

    def load_season(self, season: str) -> int:
        logger.info(f"Loading full season schedule for {season}")
        df = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            league_id_nullable='00',
            timeout=60,
        ).get_data_frames()[0]
        rows = []
        for _, row in df.iterrows():
            home_away, opp = _parse_matchup(str(row['MATCHUP']), str(row['TEAM_ABBREVIATION']))
            rows.append({
                'team_id':           int(row['TEAM_ID']),
                'team_abbreviation': str(row['TEAM_ABBREVIATION']),
                'game_id':           str(row['GAME_ID']),
                'game_date':         pd.to_datetime(row['GAME_DATE']).date(),
                'season':            season,
                'home_away':         home_away,
                'opponent_abbr':     opp,
            })
        with self.connection.get_engine().connect() as conn:
            for row in rows:
                conn.execute(_UPSERT_SQL, row)
            conn.commit()
        logger.info(f"Loaded {len(rows)} schedule rows for {season}")
        return len(rows)

    def ensure_loaded(self, season: str) -> None:
        if not self.is_loaded(season):
            self.load_season(season)

    def patch_postponements(self, game_date: date, completed_game_ids: list) -> int:
        """Mark games scheduled on game_date not in completed_game_ids as postponed."""
        sql_select = text("""
            SELECT DISTINCT game_id FROM team_schedules
            WHERE game_date = :game_date AND NOT postponed
        """)
        sql_update = text("""
            UPDATE team_schedules SET postponed = TRUE, updated_at = NOW()
            WHERE game_id = :game_id
        """)
        completed = set(completed_game_ids)
        updated = 0
        with self.connection.get_engine().connect() as conn:
            scheduled = [row[0] for row in conn.execute(sql_select, {'game_date': game_date})]
            for game_id in scheduled:
                if game_id not in completed:
                    conn.execute(sql_update, {'game_id': game_id})
                    updated += 1
            if updated:
                conn.commit()
        if updated:
            logger.info(f"Marked {updated} games as postponed on {game_date}")
        return updated
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_team_schedules.py -v
```
Expected: All 15 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add data/postgres/team_schedules.py tests/test_team_schedules.py
git commit -m "feat: add TeamScheduleManager with season load and postponement patching"
```

---

## Task 4: `opponent_defense_rankings` Table + Manager

**Files:**
- Create: `data/postgres/opponent_defense.py`
- Test: `tests/test_opponent_defense.py`

**Interfaces:**
- Produces: `OpponentDefenseManager`, `_compute_ranks(df: pd.DataFrame) -> pd.DataFrame`
- `create_table() -> bool`, `fetch_and_upsert(season: str) -> int`

**Important:** `LeagueDashTeamStats` with `measure_type_detailed_defense='Opponent'` returns columns named `OPP_PTS`, `OPP_REB`, `OPP_AST`, `OPP_STL`, `OPP_BLK`, `OPP_TOV`, `OPP_FG_PCT`, `OPP_FG3_PCT`. Verify these column names with a quick REPL call before implementing:
```python
from nba_api.stats.endpoints import leaguedashteamstats
df = leaguedashteamstats.LeagueDashTeamStats(season='2024-25', measure_type_detailed_defense='Opponent', per_mode_simple='PerGame').get_data_frames()[0]
print(df.columns.tolist())
```
If column names differ, update `_compute_ranks` accordingly.

- [ ] **Step 1: Write failing tests**

Create `tests/test_opponent_defense.py`:

```python
import pandas as pd
from unittest.mock import MagicMock, patch

from data.postgres.opponent_defense import OpponentDefenseManager, _compute_ranks


# --- _compute_ranks ---

_T1 = {
    'TEAM_ID': 1610612738, 'TEAM_ABBREVIATION': 'BOS',
    'OPP_PTS': 105.0, 'OPP_REB': 42.0, 'OPP_AST': 24.0,
    'OPP_STL': 7.0,   'OPP_BLK': 4.0,  'OPP_TOV': 14.0,
    'OPP_FG_PCT': 0.44, 'OPP_FG3_PCT': 0.33,
}
_T2 = {
    'TEAM_ID': 1610612737, 'TEAM_ABBREVIATION': 'ATL',
    'OPP_PTS': 115.0, 'OPP_REB': 46.0, 'OPP_AST': 28.0,
    'OPP_STL': 8.0,   'OPP_BLK': 5.0,  'OPP_TOV': 12.0,
    'OPP_FG_PCT': 0.48, 'OPP_FG3_PCT': 0.37,
}

def test_compute_ranks_adds_all_rank_columns():
    df = pd.DataFrame([_T1, _T2])
    result = _compute_ranks(df)
    for col in ('rank_pts', 'rank_reb', 'rank_ast', 'rank_stl', 'rank_blk', 'rank_to', 'rank_fg_pct', 'rank_3p_pct'):
        assert col in result.columns

def test_compute_ranks_pts_lower_allowed_is_rank1():
    df = pd.DataFrame([_T1, _T2])
    result = _compute_ranks(df)
    # BOS allows 105 pts (less than ATL's 115), so BOS = rank 1 (best defense)
    bos_rank = result[result['TEAM_ABBREVIATION'] == 'BOS'].iloc[0]['rank_pts']
    assert bos_rank == 1

def test_compute_ranks_to_more_forced_is_rank1():
    df = pd.DataFrame([_T1, _T2])
    result = _compute_ranks(df)
    # BOS forces 14 TOV (more than ATL's 12), so BOS = rank 1 for rank_to
    bos_rank = result[result['TEAM_ABBREVIATION'] == 'BOS'].iloc[0]['rank_to']
    assert bos_rank == 1


# --- helpers ---

def _make_manager():
    mock_conn = MagicMock()
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_pg = MagicMock()
    mock_pg.get_engine.return_value = mock_engine
    return OpponentDefenseManager(mock_pg), mock_conn


# --- create_table ---

def test_create_table_returns_true():
    mgr, _ = _make_manager()
    assert mgr.create_table() is True

def test_create_table_sql_has_opponent_defense_rankings():
    mgr, mock_conn = _make_manager()
    mgr.create_table()
    sql = mock_conn.execute.call_args[0][0].text
    assert 'opponent_defense_rankings' in sql

def test_create_table_returns_false_on_error():
    mock_pg = MagicMock()
    mock_pg.get_engine.side_effect = Exception("DB down")
    assert OpponentDefenseManager(mock_pg).create_table() is False


# --- fetch_and_upsert ---

def test_fetch_and_upsert_calls_league_dash_team_stats():
    mgr, _ = _make_manager()
    mock_ep = MagicMock()
    mock_ep.get_data_frames.return_value = [pd.DataFrame([_T1, _T2])]
    with patch('data.postgres.opponent_defense.leaguedashteamstats.LeagueDashTeamStats', return_value=mock_ep) as mock_cls:
        mgr.fetch_and_upsert('2025-26')
    mock_cls.assert_called_once()

def test_fetch_and_upsert_returns_row_count():
    mgr, _ = _make_manager()
    mock_ep = MagicMock()
    mock_ep.get_data_frames.return_value = [pd.DataFrame([_T1, _T2])]
    with patch('data.postgres.opponent_defense.leaguedashteamstats.LeagueDashTeamStats', return_value=mock_ep):
        assert mgr.fetch_and_upsert('2025-26') == 2

def test_fetch_and_upsert_sql_has_on_conflict():
    mgr, mock_conn = _make_manager()
    mock_ep = MagicMock()
    mock_ep.get_data_frames.return_value = [pd.DataFrame([_T1])]
    with patch('data.postgres.opponent_defense.leaguedashteamstats.LeagueDashTeamStats', return_value=mock_ep):
        mgr.fetch_and_upsert('2025-26')
    sql = mock_conn.execute.call_args[0][0].text
    assert 'ON CONFLICT' in sql
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_opponent_defense.py -v
```
Expected: ImportError.

- [ ] **Step 3: Verify NBA API column names (before implementing)**

Run this from the project root (requires NBA API access):
```bash
python -c "
from nba_api.stats.endpoints import leaguedashteamstats
df = leaguedashteamstats.LeagueDashTeamStats(season='2024-25', measure_type_detailed_defense='Opponent', per_mode_simple='PerGame').get_data_frames()[0]
print(df.columns.tolist())
"
```
If column names differ from `OPP_PTS`, `OPP_REB`, `OPP_AST`, `OPP_STL`, `OPP_BLK`, `OPP_TOV`, `OPP_FG_PCT`, `OPP_FG3_PCT`, update `_compute_ranks` to match.

- [ ] **Step 4: Implement `data/postgres/opponent_defense.py`**

```python
from typing import Optional

import pandas as pd
from sqlalchemy import text

from nba_api.stats.endpoints import leaguedashteamstats
from data.postgres.connection import PostgresConnection
from logger import get_logger

logger = get_logger(__name__)

_CREATE_SQL = text("""
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
    )
""")

_UPSERT_SQL = text("""
    INSERT INTO opponent_defense_rankings (
        team_id, team_abbreviation, season,
        rank_pts, rank_reb, rank_ast, rank_stl, rank_blk, rank_to,
        rank_fg_pct, rank_3p_pct, updated_at
    ) VALUES (
        :team_id, :team_abbreviation, :season,
        :rank_pts, :rank_reb, :rank_ast, :rank_stl, :rank_blk, :rank_to,
        :rank_fg_pct, :rank_3p_pct, NOW()
    )
    ON CONFLICT (team_id, season) DO UPDATE SET
        rank_pts = EXCLUDED.rank_pts, rank_reb = EXCLUDED.rank_reb,
        rank_ast = EXCLUDED.rank_ast, rank_stl = EXCLUDED.rank_stl,
        rank_blk = EXCLUDED.rank_blk, rank_to  = EXCLUDED.rank_to,
        rank_fg_pct = EXCLUDED.rank_fg_pct, rank_3p_pct = EXCLUDED.rank_3p_pct,
        updated_at = NOW()
""")


def _compute_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank 1 = best defense (fewest pts/reb/ast/stl/blk allowed).
    rank_to: rank 1 = most turnovers forced (higher OPP_TOV = better defense).
    """
    df['rank_pts']    = df['OPP_PTS'].rank(ascending=True,  method='min').astype(int)
    df['rank_reb']    = df['OPP_REB'].rank(ascending=True,  method='min').astype(int)
    df['rank_ast']    = df['OPP_AST'].rank(ascending=True,  method='min').astype(int)
    df['rank_stl']    = df['OPP_STL'].rank(ascending=True,  method='min').astype(int)
    df['rank_blk']    = df['OPP_BLK'].rank(ascending=True,  method='min').astype(int)
    df['rank_to']     = df['OPP_TOV'].rank(ascending=False, method='min').astype(int)
    df['rank_fg_pct'] = df['OPP_FG_PCT'].rank(ascending=True,  method='min').astype(int)
    df['rank_3p_pct'] = df['OPP_FG3_PCT'].rank(ascending=True, method='min').astype(int)
    return df


class OpponentDefenseManager:

    def __init__(self, connection: Optional[PostgresConnection] = None):
        self.connection = connection or PostgresConnection()

    def create_table(self) -> bool:
        try:
            with self.connection.get_engine().connect() as conn:
                conn.execute(_CREATE_SQL)
                conn.commit()
            logger.info("Created/verified opponent_defense_rankings table")
            return True
        except Exception as e:
            logger.error(f"Failed to create opponent_defense_rankings table: {e}")
            return False

    def fetch_and_upsert(self, season: str) -> int:
        logger.info(f"Fetching opponent defense rankings for {season}")
        df = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense='Opponent',
            per_mode_simple='PerGame',
            timeout=30,
        ).get_data_frames()[0]
        df = _compute_ranks(df)
        rows = []
        for _, row in df.iterrows():
            rows.append({
                'team_id':           int(row['TEAM_ID']),
                'team_abbreviation': str(row['TEAM_ABBREVIATION']),
                'season':            season,
                'rank_pts':          int(row['rank_pts']),
                'rank_reb':          int(row['rank_reb']),
                'rank_ast':          int(row['rank_ast']),
                'rank_stl':          int(row['rank_stl']),
                'rank_blk':          int(row['rank_blk']),
                'rank_to':           int(row['rank_to']),
                'rank_fg_pct':       int(row['rank_fg_pct']),
                'rank_3p_pct':       int(row['rank_3p_pct']),
            })
        with self.connection.get_engine().connect() as conn:
            for row in rows:
                conn.execute(_UPSERT_SQL, row)
            conn.commit()
        logger.info(f"Upserted {len(rows)} opponent defense ranking rows for {season}")
        return len(rows)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_opponent_defense.py -v
```
Expected: All 10 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add data/postgres/opponent_defense.py tests/test_opponent_defense.py
git commit -m "feat: add OpponentDefenseManager with weekly ranking upsert"
```

---

## Task 5: Pipeline Entry Point

**Files:**
- Create: `scripts/__init__.py` (empty)
- Create: `scripts/pipelines/__init__.py` (empty)
- Create: `scripts/pipelines/update_player_stats_db.py`
- Test: `tests/test_update_player_stats_db.py`

**Interfaces:**
- Consumes: all four managers + `is_nba_season`, `current_season` from `data.postgres.player_stats`
- Produces: `run(target_date: date = None) -> None` — the sole entry point

- [ ] **Step 1: Write failing tests**

Create `tests/test_update_player_stats_db.py`:

```python
from datetime import date
from unittest.mock import MagicMock, patch


def _make_mocks():
    game_logs = MagicMock()
    game_logs.compute_std_dev.return_value = {}
    game_logs.get_game_ids_on_date.return_value = []
    schedule = MagicMock()
    stats = MagicMock()
    stats.fetch.return_value = (MagicMock(), MagicMock(), MagicMock())
    defense = MagicMock()
    return game_logs, schedule, stats, defense


def _run(target_date, game_logs, schedule, stats, defense):
    with patch('scripts.pipelines.update_player_stats_db.PlayerGameLogsManager', return_value=game_logs):
        with patch('scripts.pipelines.update_player_stats_db.TeamScheduleManager', return_value=schedule):
            with patch('scripts.pipelines.update_player_stats_db.PlayerStatsManager', return_value=stats):
                with patch('scripts.pipelines.update_player_stats_db.OpponentDefenseManager', return_value=defense):
                    with patch('scripts.pipelines.update_player_stats_db.PostgresConnection'):
                        from scripts.pipelines.update_player_stats_db import run
                        run(target_date)


def test_run_skips_in_offseason():
    game_logs, schedule, stats, defense = _make_mocks()
    _run(date(2026, 8, 1), game_logs, schedule, stats, defense)
    game_logs.fetch_and_upsert_date.assert_not_called()

def test_run_fetches_game_logs_in_season():
    game_logs, schedule, stats, defense = _make_mocks()
    _run(date(2026, 3, 15), game_logs, schedule, stats, defense)
    game_logs.fetch_and_upsert_date.assert_called_once()

def test_run_patches_postponements():
    game_logs, schedule, stats, defense = _make_mocks()
    _run(date(2026, 3, 15), game_logs, schedule, stats, defense)
    schedule.patch_postponements.assert_called_once()

def test_run_recomputes_season_aggregates():
    game_logs, schedule, stats, defense = _make_mocks()
    _run(date(2026, 3, 15), game_logs, schedule, stats, defense)
    stats.fetch.assert_called_once()
    stats.upsert_all.assert_called_once()

def test_run_updates_std_devs():
    game_logs, schedule, stats, defense = _make_mocks()
    _run(date(2026, 3, 15), game_logs, schedule, stats, defense)
    stats.update_std_devs.assert_called_once()

def test_run_fetches_opponent_defense_on_monday():
    game_logs, schedule, stats, defense = _make_mocks()
    monday = date(2026, 3, 16)
    assert monday.weekday() == 0, "test date must be a Monday"
    _run(monday, game_logs, schedule, stats, defense)
    defense.fetch_and_upsert.assert_called_once()

def test_run_skips_opponent_defense_on_non_monday():
    game_logs, schedule, stats, defense = _make_mocks()
    tuesday = date(2026, 3, 17)
    assert tuesday.weekday() == 1, "test date must be a Tuesday"
    _run(tuesday, game_logs, schedule, stats, defense)
    defense.fetch_and_upsert.assert_not_called()

def test_run_ensures_schedule_loaded():
    game_logs, schedule, stats, defense = _make_mocks()
    _run(date(2026, 3, 15), game_logs, schedule, stats, defense)
    schedule.ensure_loaded.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_update_player_stats_db.py -v
```
Expected: ImportError — module does not exist.

- [ ] **Step 3: Create `__init__.py` files**

```bash
touch scripts/__init__.py scripts/pipelines/__init__.py
```

- [ ] **Step 4: Implement `scripts/pipelines/update_player_stats_db.py`**

```python
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.postgres.connection import PostgresConnection
from data.postgres.player_stats import PlayerStatsManager, is_nba_season, current_season
from data.postgres.player_game_logs import PlayerGameLogsManager
from data.postgres.team_schedules import TeamScheduleManager
from data.postgres.opponent_defense import OpponentDefenseManager
from logger import get_logger

logger = get_logger(__name__)


def run(target_date: date = None) -> None:
    today = target_date or date.today()

    if not is_nba_season(today):
        logger.info("Off-season — skipping player stats DB update")
        return

    yesterday = today - timedelta(days=1)
    season = current_season(today)
    conn = PostgresConnection()

    game_logs_mgr = PlayerGameLogsManager(conn)
    game_logs_mgr.create_table()
    game_logs_mgr.fetch_and_upsert_date(yesterday, season)
    completed_ids = game_logs_mgr.get_game_ids_on_date(yesterday)

    schedule_mgr = TeamScheduleManager(conn)
    schedule_mgr.create_table()
    schedule_mgr.ensure_loaded(season)
    schedule_mgr.patch_postponements(yesterday, completed_ids)

    stats_mgr = PlayerStatsManager(conn)
    pg, tot, p36 = stats_mgr.fetch(season)
    stats_mgr.upsert_all(pg, tot, p36, season)

    std_devs = game_logs_mgr.compute_std_dev(season)
    stats_mgr.update_std_devs(std_devs, season)

    if today.weekday() == 0:
        defense_mgr = OpponentDefenseManager(conn)
        defense_mgr.create_table()
        defense_mgr.fetch_and_upsert(season)

    logger.info("Player stats DB update complete")


if __name__ == '__main__':
    run()
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_update_player_stats_db.py -v
```
Expected: All 8 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/__init__.py scripts/pipelines/__init__.py scripts/pipelines/update_player_stats_db.py tests/test_update_player_stats_db.py
git commit -m "feat: add update_player_stats_db pipeline entry point with nightly + weekly steps"
```

---

## Task 6: Agent Tools

**Files:**
- Create: `agents/tools/player_stats_db.py`
- Test: `tests/test_player_stats_db_tools.py`

**Interfaces:**
- Produces: `PlayerStatsDbTool(connection=None, debug=False)` with `get_all_tools() -> list`
- Tool names (exactly): `get_recent_form`, `get_schedule_density`, `get_season_trends`, `query_stats_db`
- `_is_safe_sql(sql: str) -> bool`

- [ ] **Step 1: Write failing tests**

Create `tests/test_player_stats_db_tools.py`:

```python
import json
from datetime import date
from unittest.mock import MagicMock

from agents.tools.player_stats_db import PlayerStatsDbTool, _is_safe_sql


# --- _is_safe_sql ---

def test_is_safe_sql_allows_select():
    assert _is_safe_sql('SELECT * FROM player_game_logs') is True

def test_is_safe_sql_allows_select_with_leading_whitespace():
    assert _is_safe_sql('  SELECT player_id FROM player_game_logs') is True

def test_is_safe_sql_rejects_insert():
    assert _is_safe_sql('INSERT INTO player_game_logs VALUES (1)') is False

def test_is_safe_sql_rejects_update():
    assert _is_safe_sql('UPDATE player_game_logs SET pts=0') is False

def test_is_safe_sql_rejects_drop():
    assert _is_safe_sql('DROP TABLE player_game_logs') is False

def test_is_safe_sql_rejects_delete():
    assert _is_safe_sql('DELETE FROM player_game_logs') is False


# --- helpers ---

def _make_tools():
    mock_ctx = MagicMock()
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_ctx
    mock_pg = MagicMock()
    mock_pg.get_engine.return_value = mock_engine
    tool_obj = PlayerStatsDbTool(connection=mock_pg)
    tools = {t.name: t for t in tool_obj.get_all_tools()}
    return tools, mock_ctx


# --- get_all_tools ---

def test_get_all_tools_returns_four_tools():
    tools, _ = _make_tools()
    assert len(tools) == 4

def test_get_all_tools_has_expected_names():
    tools, _ = _make_tools()
    assert 'get_recent_form' in tools
    assert 'get_schedule_density' in tools
    assert 'get_season_trends' in tools
    assert 'query_stats_db' in tools


# --- get_recent_form ---

def test_get_recent_form_returns_json_with_recent_and_season_stats():
    tools, mock_ctx = _make_tools()
    keys = [
        'recent_games', 'recent_pts', 'recent_reb', 'recent_ast',
        'recent_stl', 'recent_blk', 'recent_tov', 'recent_fg_pct', 'recent_3p_pct', 'recent_ft_pct',
        'season_pts', 'season_reb', 'season_ast', 'season_stl', 'season_blk', 'season_tov',
        'season_fg_pct', 'season_3p_pct', 'season_ft_pct',
    ]
    mock_ctx.execute.return_value.fetchone.return_value = (10, 25.0, 7.0, 8.5, 1.2, 0.5, 3.0, 0.510, 0.360, 0.740,
                                                            24.0, 7.2, 8.1, 1.1, 0.4, 3.2, 0.500, 0.350, 0.730)
    mock_ctx.execute.return_value.keys.return_value = keys
    result = json.loads(tools['get_recent_form'].invoke({'player_id': 2544}))
    assert result['recent_pts'] == 25.0
    assert result['season_pts'] == 24.0

def test_get_recent_form_returns_error_for_missing_player():
    tools, mock_ctx = _make_tools()
    mock_ctx.execute.return_value.fetchone.return_value = (0,) + (None,) * 18
    mock_ctx.execute.return_value.keys.return_value = ['recent_games'] + ['x'] * 18
    result = json.loads(tools['get_recent_form'].invoke({'player_id': 99999}))
    assert 'error' in result

def test_get_recent_form_default_n_games_is_10():
    tools, mock_ctx = _make_tools()
    mock_ctx.execute.return_value.fetchone.return_value = (0,) + (None,) * 18
    mock_ctx.execute.return_value.keys.return_value = ['recent_games'] + ['x'] * 18
    tools['get_recent_form'].invoke({'player_id': 2544})
    params = mock_ctx.execute.call_args[0][1]
    assert params['n_games'] == 10


# --- get_schedule_density ---

def test_get_schedule_density_returns_game_list():
    tools, mock_ctx = _make_tools()
    mock_ctx.execute.return_value.__iter__ = lambda s: iter([
        (date(2026, 3, 16), 'MIA', 'home'),
        (date(2026, 3, 18), 'BKN', 'away'),
    ])
    result = json.loads(tools['get_schedule_density'].invoke({'team_abbreviation': 'BOS'}))
    assert result['game_count'] == 2
    assert result['games'][0]['opponent'] == 'MIA'

def test_get_schedule_density_default_days_is_7():
    tools, mock_ctx = _make_tools()
    mock_ctx.execute.return_value.__iter__ = lambda s: iter([])
    tools['get_schedule_density'].invoke({'team_abbreviation': 'BOS'})
    params = mock_ctx.execute.call_args[0][1]
    assert params['days'] == 7


# --- get_season_trends ---

def test_get_season_trends_returns_monthly_trends():
    tools, mock_ctx = _make_tools()
    keys = ['month', 'games', 'pts', 'reb', 'ast', 'stl', 'blk', 'tov', 'fg_pct', 'fg3_pct', 'ft_pct']
    mock_ctx.execute.return_value.fetchall.return_value = [
        ('2025-10', 10, 24.0, 7.0, 8.0, 1.1, 0.4, 3.0, 0.50, 0.35, 0.73)
    ]
    mock_ctx.execute.return_value.keys.return_value = keys
    result = json.loads(tools['get_season_trends'].invoke({'player_id': 2544}))
    assert len(result['monthly_trends']) == 1
    assert result['monthly_trends'][0]['month'] == '2025-10'

def test_get_season_trends_returns_error_for_missing_player():
    tools, mock_ctx = _make_tools()
    mock_ctx.execute.return_value.fetchall.return_value = []
    result = json.loads(tools['get_season_trends'].invoke({'player_id': 99999}))
    assert 'error' in result


# --- query_stats_db ---

def test_query_stats_db_executes_select_and_returns_rows():
    tools, mock_ctx = _make_tools()
    mock_ctx.execute.return_value.fetchall.return_value = [(2544, 'LeBron James')]
    mock_ctx.execute.return_value.keys.return_value = ['player_id', 'player_name']
    result = json.loads(tools['query_stats_db'].invoke({'sql': 'SELECT player_id, player_name FROM player_game_logs LIMIT 1'}))
    assert result[0]['player_id'] == 2544

def test_query_stats_db_rejects_drop_without_executing():
    tools, mock_ctx = _make_tools()
    result = json.loads(tools['query_stats_db'].invoke({'sql': 'DROP TABLE player_game_logs'}))
    assert 'error' in result
    mock_ctx.execute.assert_not_called()

def test_query_stats_db_rejects_insert_without_executing():
    tools, mock_ctx = _make_tools()
    result = json.loads(tools['query_stats_db'].invoke({'sql': 'INSERT INTO player_game_logs VALUES (1)'}))
    assert 'error' in result
    mock_ctx.execute.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_player_stats_db_tools.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `agents/tools/player_stats_db.py`**

```python
import json
from typing import Optional

from langchain_core.tools import tool
from sqlalchemy import text

from agents.tools.base import ClutchAITool
from data.postgres.connection import PostgresConnection
from logger import get_logger

logger = get_logger(__name__)


def _is_safe_sql(sql: str) -> bool:
    return sql.strip().lower().startswith('select')


class PlayerStatsDbTool(ClutchAITool):
    """DB-backed tools for StatsAgent: recent form, schedule density, season trends, raw SQL."""

    def __init__(self, connection: Optional[PostgresConnection] = None, debug: bool = False):
        super().__init__(debug=debug)
        self.connection = connection or PostgresConnection()

    def get_all_tools(self) -> list:
        connection = self.connection

        @tool
        def get_recent_form(player_id: int, n_games: int = 10) -> str:
            """
            Get a player's recent form from the last N games (default 10).
            Returns per-game averages (PTS, REB, AST, STL, BLK, TOV, FG%, 3P%, FT%)
            for the last N games vs the season average. Use for hot/cold streak detection.

            Args:
                player_id: NBA player ID
                n_games: Number of recent games to average (default 10)
            """
            sql = text("""
                WITH recent_stats AS (
                    SELECT
                        COUNT(*)                           AS games,
                        ROUND(AVG(pts)::numeric, 1)       AS avg_pts,
                        ROUND(AVG(reb)::numeric, 1)       AS avg_reb,
                        ROUND(AVG(ast)::numeric, 1)       AS avg_ast,
                        ROUND(AVG(stl)::numeric, 1)       AS avg_stl,
                        ROUND(AVG(blk)::numeric, 1)       AS avg_blk,
                        ROUND(AVG("to")::numeric, 1)      AS avg_tov,
                        ROUND(AVG(fg_pct)::numeric, 3)    AS avg_fg_pct,
                        ROUND(AVG(fg3_pct)::numeric, 3)   AS avg_3p_pct,
                        ROUND(AVG(ft_pct)::numeric, 3)    AS avg_ft_pct
                    FROM (
                        SELECT * FROM player_game_logs
                        WHERE player_id = :player_id
                        ORDER BY game_date DESC
                        LIMIT :n_games
                    ) r
                ),
                season_stats AS (
                    SELECT
                        ROUND(AVG(pts)::numeric, 1)       AS avg_pts,
                        ROUND(AVG(reb)::numeric, 1)       AS avg_reb,
                        ROUND(AVG(ast)::numeric, 1)       AS avg_ast,
                        ROUND(AVG(stl)::numeric, 1)       AS avg_stl,
                        ROUND(AVG(blk)::numeric, 1)       AS avg_blk,
                        ROUND(AVG("to")::numeric, 1)      AS avg_tov,
                        ROUND(AVG(fg_pct)::numeric, 3)    AS avg_fg_pct,
                        ROUND(AVG(fg3_pct)::numeric, 3)   AS avg_3p_pct,
                        ROUND(AVG(ft_pct)::numeric, 3)    AS avg_ft_pct
                    FROM player_game_logs
                    WHERE player_id = :player_id
                )
                SELECT
                    rs.games                AS recent_games,
                    rs.avg_pts              AS recent_pts,
                    rs.avg_reb              AS recent_reb,
                    rs.avg_ast              AS recent_ast,
                    rs.avg_stl              AS recent_stl,
                    rs.avg_blk              AS recent_blk,
                    rs.avg_tov              AS recent_tov,
                    rs.avg_fg_pct           AS recent_fg_pct,
                    rs.avg_3p_pct           AS recent_3p_pct,
                    rs.avg_ft_pct           AS recent_ft_pct,
                    ss.avg_pts              AS season_pts,
                    ss.avg_reb              AS season_reb,
                    ss.avg_ast              AS season_ast,
                    ss.avg_stl              AS season_stl,
                    ss.avg_blk              AS season_blk,
                    ss.avg_tov              AS season_tov,
                    ss.avg_fg_pct           AS season_fg_pct,
                    ss.avg_3p_pct           AS season_3p_pct,
                    ss.avg_ft_pct           AS season_ft_pct
                FROM recent_stats rs, season_stats ss
            """)
            try:
                with connection.get_engine().connect() as conn:
                    result = conn.execute(sql, {'player_id': player_id, 'n_games': n_games})
                    row = result.fetchone()
                    if not row or row[0] == 0:
                        return json.dumps({'error': f'No game logs found for player_id={player_id}'})
                    return json.dumps(dict(zip(result.keys(), row)), default=str)
            except Exception as e:
                logger.error(f"get_recent_form error: {e}")
                return json.dumps({'error': str(e)})

        @tool
        def get_schedule_density(team_abbreviation: str, days: int = 7) -> str:
            """
            Get upcoming games for a team in the next N days (default 7).
            Returns game list (date, opponent, home/away) plus total count.
            Use for streaming pickup recommendations and start/sit decisions.

            Args:
                team_abbreviation: NBA team abbreviation (e.g., 'LAL', 'BOS')
                days: Number of days to look ahead (default 7)
            """
            sql = text("""
                SELECT game_date, opponent_abbr, home_away
                FROM team_schedules
                WHERE team_abbreviation = :team_abbr
                  AND game_date >= CURRENT_DATE
                  AND game_date < CURRENT_DATE + :days * INTERVAL '1 day'
                  AND NOT postponed
                ORDER BY game_date
            """)
            try:
                with connection.get_engine().connect() as conn:
                    result = conn.execute(sql, {'team_abbr': team_abbreviation, 'days': days})
                    games = [
                        {'date': str(row[0]), 'opponent': row[1], 'home_away': row[2]}
                        for row in result
                    ]
                return json.dumps({
                    'team': team_abbreviation,
                    'days_ahead': days,
                    'game_count': len(games),
                    'games': games,
                }, default=str)
            except Exception as e:
                logger.error(f"get_schedule_density error: {e}")
                return json.dumps({'error': str(e)})

        @tool
        def get_season_trends(player_id: int) -> str:
            """
            Get a player's monthly stat trends over the current season.
            Returns per-game averages by calendar month showing production trajectory.
            Use for buy-low/sell-high analysis and identifying improving/declining players.

            Args:
                player_id: NBA player ID
            """
            sql = text("""
                SELECT
                    TO_CHAR(game_date, 'YYYY-MM')      AS month,
                    COUNT(*)                            AS games,
                    ROUND(AVG(pts)::numeric, 1)        AS pts,
                    ROUND(AVG(reb)::numeric, 1)        AS reb,
                    ROUND(AVG(ast)::numeric, 1)        AS ast,
                    ROUND(AVG(stl)::numeric, 1)        AS stl,
                    ROUND(AVG(blk)::numeric, 1)        AS blk,
                    ROUND(AVG("to")::numeric, 1)       AS tov,
                    ROUND(AVG(fg_pct)::numeric, 3)     AS fg_pct,
                    ROUND(AVG(fg3_pct)::numeric, 3)    AS fg3_pct,
                    ROUND(AVG(ft_pct)::numeric, 3)     AS ft_pct
                FROM player_game_logs
                WHERE player_id = :player_id
                GROUP BY TO_CHAR(game_date, 'YYYY-MM')
                ORDER BY month
            """)
            try:
                with connection.get_engine().connect() as conn:
                    result = conn.execute(sql, {'player_id': player_id})
                    rows = result.fetchall()
                    if not rows:
                        return json.dumps({'error': f'No game logs found for player_id={player_id}'})
                    keys = result.keys()
                    return json.dumps(
                        {'player_id': player_id, 'monthly_trends': [dict(zip(keys, r)) for r in rows]},
                        default=str,
                    )
            except Exception as e:
                logger.error(f"get_season_trends error: {e}")
                return json.dumps({'error': str(e)})

        @tool
        def query_stats_db(sql: str) -> str:
            """
            Execute a read-only SELECT query against the player stats database.

            Available tables:
            - player_game_logs(player_id, game_id, game_date, season, team_abbreviation,
                                min, pts, reb, ast, stl, blk, "to", fgm, fga, fg_pct,
                                fg3m, fg3a, fg3_pct, ftm, fta, ft_pct, plus_minus)
            - team_schedules(team_id, team_abbreviation, game_id, game_date, season,
                              home_away, opponent_abbr, postponed)
            - opponent_defense_rankings(team_id, team_abbreviation, season,
                                         rank_pts, rank_reb, rank_ast, rank_stl, rank_blk,
                                         rank_to, rank_fg_pct, rank_3p_pct)
                                         Rank 1=best defense (hardest), 30=worst (easiest matchup).
            - bball_monsters_player_stats_pg / _total / _p36 — season aggregates with z-scores.

            Only SELECT statements are allowed.

            Args:
                sql: A SQL SELECT statement
            """
            if not _is_safe_sql(sql):
                return json.dumps({'error': 'Only SELECT statements are allowed.'})
            try:
                with connection.get_engine().connect() as conn:
                    result = conn.execute(text(sql))
                    rows = result.fetchall()
                    return json.dumps([dict(zip(result.keys(), r)) for r in rows], default=str)
            except Exception as e:
                logger.error(f"query_stats_db error: {e}")
                return json.dumps({'error': str(e)})

        return [get_recent_form, get_schedule_density, get_season_trends, query_stats_db]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_player_stats_db_tools.py -v
```
Expected: All 17 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add agents/tools/player_stats_db.py tests/test_player_stats_db_tools.py
git commit -m "feat: add PlayerStatsDbTool with get_recent_form, get_schedule_density, get_season_trends, query_stats_db"
```

---

## Task 7: Wire Tools Into StatisticAgent

**Files:**
- Modify: `agents/multi_agent/statistic_agent.py`

**Interfaces:**
- Consumes: `PlayerStatsDbTool` from `agents.tools.player_stats_db`
- No new tests needed — the tool class is tested in Task 6; agent loading follows established pattern

- [ ] **Step 1: Update `_get_default_system_prompt` in `agents/multi_agent/statistic_agent.py`**

Replace the existing `_get_default_system_prompt` method body:

```python
def _get_default_system_prompt(self) -> str:
    return """You are a statistics specialist for fantasy basketball analysis.

Use basketball_monster_stats for player stats with value metrics:
- stat_type='pg'    → per-game averages + z-scores + rv, pv, three_v
- stat_type='total' → season totals + z-scores + rv, three_v
- stat_type='p36'   → per-36 minute rates + z-scores + rv, pv, three_v
Available seasons: 2023-24, 2024-25, 2025-26.

Z-scores: pV, rV, aV, sV, bV, pts3V (=three_v), toV, fgV, ftV.
value = sum of all 9 z-scores. pv = Yahoo points league value vs replacement.

For in-season analysis, prefer DB tools over live NBA API tools:
- get_recent_form(player_id, n_games=10) — hot/cold streaks, waiver wire form
- get_schedule_density(team_abbreviation, days=7) — games remaining, matchup density
- get_season_trends(player_id) — monthly trajectory, buy-low/sell-high
- query_stats_db(sql) — custom cross-table queries

opponent_defense_rankings ranks: 1=best defense (hardest matchup), 30=worst defense (easiest matchup).

Use live NBA API tools only for data not in the DB: live box scores, play-by-play, career stats."""
```

- [ ] **Step 2: Add `PlayerStatsDbTool` to `_create_tools` in the same file**

In `_create_tools`, after the `PlayerStatsTool` block and before the `nbaAPITool` block, add:

```python
try:
    from agents.tools.player_stats_db import PlayerStatsDbTool
    tools.extend(PlayerStatsDbTool(debug=self.debug).get_all_tools())
    self.logger.debug("PlayerStatsDbTool loaded")
except Exception as e:
    self.logger.warning(f"PlayerStatsDbTool not available: {e}")
```

- [ ] **Step 3: Run full test suite to verify no regressions**

```bash
pytest tests/ -v --ignore=tests/test_nba_api_integration.py --ignore=tests/test_yahoo_api_integration.py --ignore=tests/test_rag_integration.py
```
Expected: All unit tests PASS. (Integration tests excluded — they require live API/DB access.)

- [ ] **Step 4: Commit**

```bash
git add agents/multi_agent/statistic_agent.py
git commit -m "feat: add PlayerStatsDbTool to StatisticAgent with updated routing system prompt"
```

---

## Task 8: Railway `player-db-cron` Service

This task uses Railway MCP tools — no code files involved.

- [ ] **Step 1: Create the Railway service via MCP**

Using `mcp__railway__create_service` for project `16236bdb-073d-4853-9021-5a8e2af0eea4`:
- Service name: `player-db-cron`
- Start command: `bash -c "python scripts/pipelines/update_player_stats_db.py"`
- Cron schedule: `0 6 * * *` (2am ET = 6am UTC)

- [ ] **Step 2: Set required environment variables**

Using `mcp__railway__set_variables` — the service inherits `DATABASE_URL` and `OPENAI_API_KEY` from the Railway project shared variables. Verify both are available; if not, add them explicitly for this service.

- [ ] **Step 3: Verify service status**

Using `mcp__railway__environment_status` — confirm `player-db-cron` appears with status `SUCCESS`.

- [ ] **Step 4: Run schema migration in production**

SSH into the Railway service or trigger a one-off run to execute the std dev column migration. The simplest approach: temporarily override the start command to run the migration, then restore it.

Alternatively, add `migrate_std_dev_cols` to the pipeline's `run()` function so it runs on every invocation (idempotent — `ADD COLUMN IF NOT EXISTS`).

**Recommended:** Add the migration call to `run()` in `scripts/pipelines/update_player_stats_db.py` before the daily steps:

```python
from data.postgres.player_stats import PlayerStatsManager, is_nba_season, current_season, migrate_std_dev_cols

def run(target_date: date = None) -> None:
    ...
    conn = PostgresConnection()
    migrate_std_dev_cols(conn)  # idempotent; safe to run every time
    ...
```

Commit this change, then redeploy.

- [ ] **Step 5: Final commit and push**

```bash
git add scripts/pipelines/update_player_stats_db.py
git commit -m "feat: run std dev migration on every pipeline start (idempotent)"
git push -u origin feature/player_db
```
