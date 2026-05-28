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
               'stl': 3.0, 'blk': 3.0, 'to': -1.0}

_READ_SQL = lambda table, season: text(f"""
    SELECT player_id, season, min, pts, reb, ast, stl, blk, "to",
           fg3m, fg_pct, fga, ft_pct, fta
    FROM {table}
    WHERE season = :season
""")

# z-score cols quoted to preserve case; aggregate renamed to value
_WRITE_SQL = lambda table: text(f"""
    UPDATE {table}
    SET "pV"    = :pV,    "rV"    = :rV,    "aV"    = :aV,
        "sV"    = :sV,    "bV"    = :bV,    "pts3V" = :pts3V,
        "toV"   = :toV,   "fgV"   = :fgV,   "ftV"   = :ftV,
        value   = :value, three_v = :three_v,
        updated_at = NOW()
    WHERE player_id = :player_id AND season = :season
""")

_WRITE_WITH_PV_SQL = lambda table: text(f"""
    UPDATE {table}
    SET "pV"    = :pV,    "rV"    = :rV,    "aV"    = :aV,
        "sV"    = :sV,    "bV"    = :bV,    "pts3V" = :pts3V,
        "toV"   = :toV,   "fgV"   = :fgV,   "ftV"   = :ftV,
        value   = :value, three_v = :three_v, pv     = :pv,
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

        df['pV']    = _z('pts')
        df['rV']    = _z('reb')
        df['aV']    = _z('ast')
        df['sV']    = _z('stl')
        df['bV']    = _z('blk')
        df['pts3V'] = _z('fg3m')
        df['toV']   = _z('to', negate=True)
        df['fgV']   = _z_vw('fg_pct', 'fga')
        df['ftV']   = _z_vw('ft_pct', 'fta')

        z_cols = ['pV', 'rV', 'aV', 'sV', 'bV', 'pts3V', 'toV', 'fgV', 'ftV']
        df['value']   = df[z_cols].sum(axis=1)
        df['three_v'] = df['pts3V']

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
                    'pV':    row['pV'],    'rV':    row['rV'],
                    'aV':    row['aV'],    'sV':    row['sV'],
                    'bV':    row['bV'],    'pts3V': row['pts3V'],
                    'toV':   row['toV'],   'fgV':   row['fgV'],
                    'ftV':   row['ftV'],   'value': row['value'],
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

        # Pool is always anchored by pg table min column
        pg_df = self._read_table(_PG_TABLE, season)

        for key, (tbl, include_pv) in to_run.items():
            df = pg_df if key == 'pg' else self._read_table(tbl, season)
            if key != 'pg':
                df = df.merge(pg_df[['player_id', 'min']], on='player_id',
                              suffixes=('_orig', ''), how='left')
            result = self._compute_values(df, include_pv=include_pv)
            self._write_values(result, tbl, include_pv=include_pv)
            logger.info(f"  Calculated values for {tbl} ({len(result)} players)")
