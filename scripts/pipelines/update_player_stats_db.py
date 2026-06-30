import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.postgres.connection import PostgresConnection
from data.postgres.player_stats import PlayerStatsManager, is_nba_season, current_season, migrate_std_dev_cols
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
    migrate_std_dev_cols(conn)

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
