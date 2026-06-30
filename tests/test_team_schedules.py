import pandas as pd
from datetime import date
from unittest.mock import MagicMock, patch

from data.postgres.team_schedules import TeamScheduleManager, _parse_matchup


# --- _parse_matchup ---

def test_parse_matchup_home_game():
    home_away, opp = _parse_matchup('BOS vs. MIA', 'BOS')
    assert home_away == 'home'
    assert opp == 'MIA'

def test_parse_matchup_away_game():
    home_away, opp = _parse_matchup('BOS @ MIA', 'BOS')
    assert home_away == 'away'
    assert opp == 'MIA'


# --- helpers ---

def _make_manager():
    mock_conn = MagicMock()
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_pg = MagicMock()
    mock_pg.get_engine.return_value = mock_engine
    return TeamScheduleManager(mock_pg), mock_conn


_SCHEDULE_ROW = {
    'TEAM_ID': 1610612738, 'TEAM_ABBREVIATION': 'BOS',
    'GAME_ID': '0022500001', 'GAME_DATE': '2025-10-22',
    'MATCHUP': 'BOS vs. MIA', 'WL': 'W',
}


# --- create_table ---

def test_create_table_returns_true():
    mgr, _ = _make_manager()
    assert mgr.create_table() is True

def test_create_table_sql_has_team_schedules():
    mgr, mock_conn = _make_manager()
    mgr.create_table()
    sql = mock_conn.execute.call_args[0][0].text
    assert 'team_schedules' in sql

def test_create_table_returns_false_on_error():
    mock_pg = MagicMock()
    mock_pg.get_engine.side_effect = Exception("DB down")
    assert TeamScheduleManager(mock_pg).create_table() is False


# --- is_loaded ---

def test_is_loaded_true_when_rows_exist():
    mgr, mock_conn = _make_manager()
    mock_conn.execute.return_value.scalar.return_value = 100
    assert mgr.is_loaded('2025-26') is True

def test_is_loaded_false_when_no_rows():
    mgr, mock_conn = _make_manager()
    mock_conn.execute.return_value.scalar.return_value = 0
    assert mgr.is_loaded('2025-26') is False


# --- load_season ---

def test_load_season_calls_league_game_finder():
    mgr, _ = _make_manager()
    mock_lgf = MagicMock()
    mock_lgf.get_data_frames.return_value = [pd.DataFrame([_SCHEDULE_ROW])]
    with patch('data.postgres.team_schedules.leaguegamefinder.LeagueGameFinder', return_value=mock_lgf) as mock_cls:
        mgr.load_season('2025-26')
    mock_cls.assert_called_once()

def test_load_season_returns_row_count():
    mgr, _ = _make_manager()
    mock_lgf = MagicMock()
    mock_lgf.get_data_frames.return_value = [pd.DataFrame([_SCHEDULE_ROW, _SCHEDULE_ROW])]
    with patch('data.postgres.team_schedules.leaguegamefinder.LeagueGameFinder', return_value=mock_lgf):
        assert mgr.load_season('2025-26') == 2

def test_load_season_sql_has_on_conflict():
    mgr, mock_conn = _make_manager()
    mock_lgf = MagicMock()
    mock_lgf.get_data_frames.return_value = [pd.DataFrame([_SCHEDULE_ROW])]
    with patch('data.postgres.team_schedules.leaguegamefinder.LeagueGameFinder', return_value=mock_lgf):
        mgr.load_season('2025-26')
    sql = mock_conn.execute.call_args[0][0].text
    assert 'ON CONFLICT' in sql


# --- ensure_loaded ---

def test_ensure_loaded_calls_load_season_when_empty():
    mgr, mock_conn = _make_manager()
    mock_conn.execute.return_value.scalar.return_value = 0
    mock_lgf = MagicMock()
    mock_lgf.get_data_frames.return_value = [pd.DataFrame(columns=list(_SCHEDULE_ROW.keys()))]
    with patch('data.postgres.team_schedules.leaguegamefinder.LeagueGameFinder', return_value=mock_lgf) as mock_cls:
        mgr.ensure_loaded('2025-26')
    mock_cls.assert_called_once()

def test_ensure_loaded_skips_load_when_already_loaded():
    mgr, mock_conn = _make_manager()
    mock_conn.execute.return_value.scalar.return_value = 1000
    with patch('data.postgres.team_schedules.leaguegamefinder.LeagueGameFinder') as mock_cls:
        mgr.ensure_loaded('2025-26')
    mock_cls.assert_not_called()


# --- patch_postponements ---

def test_patch_postponements_marks_missing_game():
    mgr, mock_conn = _make_manager()
    mock_conn.execute.return_value = iter([('0022500001',), ('0022500002',)])
    updated = mgr.patch_postponements(date(2026, 3, 15), completed_game_ids=['0022500001'])
    assert updated == 1

def test_patch_postponements_returns_zero_when_all_completed():
    mgr, mock_conn = _make_manager()
    mock_conn.execute.return_value = iter([('0022500001',)])
    updated = mgr.patch_postponements(date(2026, 3, 15), completed_game_ids=['0022500001'])
    assert updated == 0

def test_patch_postponements_returns_zero_when_no_scheduled_games():
    mgr, mock_conn = _make_manager()
    mock_conn.execute.return_value = iter([])
    assert mgr.patch_postponements(date(2026, 7, 4), completed_game_ids=[]) == 0
