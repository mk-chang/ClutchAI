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
                        with patch('scripts.pipelines.update_player_stats_db.migrate_std_dev_cols'):
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

def test_run_migrates_std_dev_schema():
    game_logs, schedule, stats, defense = _make_mocks()
    with patch('scripts.pipelines.update_player_stats_db.PlayerGameLogsManager', return_value=game_logs), \
         patch('scripts.pipelines.update_player_stats_db.TeamScheduleManager', return_value=schedule), \
         patch('scripts.pipelines.update_player_stats_db.PlayerStatsManager', return_value=stats), \
         patch('scripts.pipelines.update_player_stats_db.OpponentDefenseManager', return_value=defense), \
         patch('scripts.pipelines.update_player_stats_db.PostgresConnection'), \
         patch('scripts.pipelines.update_player_stats_db.migrate_std_dev_cols') as mock_migrate:
        from scripts.pipelines.update_player_stats_db import run
        run(date(2026, 3, 15))
    mock_migrate.assert_called_once()
