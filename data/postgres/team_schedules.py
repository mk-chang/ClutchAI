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


def _parse_matchup(matchup: str) -> tuple:
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
            home_away, opp = _parse_matchup(str(row['MATCHUP']))
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
