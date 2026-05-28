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
_P36_STAT_COLS = [c for c in _STAT_COLS if c != 'PLUS_MINUS']
_INFO_COLS = ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'AGE', 'GP']

# BBM-style names; quoted in SQL to preserve case (avoids collision with pv/to stat cols)
_Z_COLS = ['pV', 'rV', 'aV', 'sV', 'bV', 'pts3V', 'toV', 'fgV', 'ftV']
_PG_VALUE_COLS  = ['value', 'three_v', 'pv']
_TOT_VALUE_COLS = ['value', 'three_v']
_P36_VALUE_COLS = ['value', 'three_v', 'pv']

# NBA API col (lowercase) → SQL col name overrides
_STAT_SQL_RENAMES = {'tov': 'to'}


def _stat_sql_col(nba_col: str) -> str:
    return _STAT_SQL_RENAMES.get(nba_col.lower(), nba_col.lower())


def _make_create_sql(table: str, stat_cols: list, value_cols: list) -> text:
    stat_ddl  = ',\n        '.join(f'{_stat_sql_col(c)} FLOAT' for c in stat_cols)
    z_ddl     = ',\n        '.join(f'"{c}" FLOAT' for c in _Z_COLS)
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
    sql_stat  = [_stat_sql_col(c) for c in stat_cols]
    all_value = list(value_cols)

    info_cols     = ['player_id', 'season', 'player_name', 'team_abbreviation', 'age', 'gp']
    z_col_sql     = [f'"{c}"' for c in _Z_COLS]

    insert_cols   = info_cols + sql_stat + z_col_sql + all_value
    insert_params = ([f':{c}' for c in info_cols + sql_stat] +
                     [f':{c}' for c in _Z_COLS] +
                     [f':{c}' for c in all_value])

    update_parts  = (
        [f'{c} = EXCLUDED.{c}' for c in ['player_name', 'team_abbreviation', 'age', 'gp'] + sql_stat] +
        [f'"{c}" = EXCLUDED."{c}"' for c in _Z_COLS] +
        [f'{c} = EXCLUDED.{c}' for c in all_value]
    )

    return text(f"""
        INSERT INTO {table} (
            {', '.join(insert_cols)},
            updated_at
        ) VALUES (
            {', '.join(insert_params)},
            NOW()
        )
        ON CONFLICT (player_id, season) DO UPDATE SET
            {', '.join(update_parts)},
            updated_at = NOW()
    """)


_PG_TABLE  = 'bball_monsters_player_stats_pg'
_TOT_TABLE = 'bball_monsters_player_stats_total'
_P36_TABLE = 'bball_monsters_player_stats_p36'

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
                    params[_stat_sql_col(c)] = row.get(c)
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
