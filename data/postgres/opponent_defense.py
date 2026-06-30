from typing import Optional

import pandas as pd
from sqlalchemy import text

from nba_api.stats.endpoints import leaguedashteamstats
from nba_api.stats.static import teams as nba_teams
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


def _get_team_abbreviation_map():
    """Build a mapping from TEAM_ID to TEAM_ABBREVIATION."""
    all_teams = nba_teams.get_teams()
    return {team['id']: team['abbreviation'] for team in all_teams}


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
            timeout=30,
        ).get_data_frames()[0]
        df = _compute_ranks(df)
        team_abbr_map = _get_team_abbreviation_map()
        rows = []
        for _, row in df.iterrows():
            team_id = int(row['TEAM_ID'])
            rows.append({
                'team_id':           team_id,
                'team_abbreviation': team_abbr_map.get(team_id, ''),
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
