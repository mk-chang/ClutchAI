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
