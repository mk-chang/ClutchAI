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
