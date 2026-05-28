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
