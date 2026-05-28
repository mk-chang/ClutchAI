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
