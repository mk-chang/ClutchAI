import math
import pandas as pd
from datetime import date
from unittest.mock import MagicMock, patch

from data.postgres.player_game_logs import PlayerGameLogsManager, _parse_min


# --- _parse_min ---

def test_parse_min_mm_ss_string():
    assert abs(_parse_min('35:23') - (35 + 23/60)) < 0.001

def test_parse_min_float_passthrough():
    assert _parse_min(35.5) == 35.5

def test_parse_min_int_passthrough():
    assert _parse_min(32) == 32.0

def test_parse_min_none_returns_none():
    assert _parse_min(None) is None

def test_parse_min_nan_returns_none():
    assert _parse_min(float('nan')) is None


# --- helpers ---

def _make_manager():
    mock_conn = MagicMock()
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_pg = MagicMock()
    mock_pg.get_engine.return_value = mock_engine
    return PlayerGameLogsManager(mock_pg), mock_conn


_BOX_ROW = {
    'PLAYER_ID': 2544, 'TEAM_ABBREVIATION': 'LAL',
    'MIN': '35:00', 'PTS': 28.0, 'REB': 7.0, 'AST': 9.0,
    'STL': 1.0, 'BLK': 0.0, 'TO': 3.0,
    'FGM': 10.0, 'FGA': 20.0, 'FG_PCT': 0.500,
    'FG3M': 2.0, 'FG3A': 6.0, 'FG3_PCT': 0.333,
    'FTM': 6.0, 'FTA': 8.0, 'FT_PCT': 0.750, 'PLUS_MINUS': 5.0,
}


# --- create_table ---

def test_create_table_returns_true():
    mgr, _ = _make_manager()
    assert mgr.create_table() is True

def test_create_table_sql_has_player_game_logs():
    mgr, mock_conn = _make_manager()
    mgr.create_table()
    sql = mock_conn.execute.call_args[0][0].text
    assert 'player_game_logs' in sql

def test_create_table_returns_false_on_error():
    mock_pg = MagicMock()
    mock_pg.get_engine.side_effect = Exception("DB down")
    assert PlayerGameLogsManager(mock_pg).create_table() is False


# --- _get_game_ids ---

def test_get_game_ids_returns_list():
    mgr, _ = _make_manager()
    mock_sb = MagicMock()
    mock_sb.get_data_frames.return_value = [pd.DataFrame({'GAME_ID': ['0022500001', '0022500002']})]
    with patch('data.postgres.player_game_logs.scoreboardv2.ScoreboardV2', return_value=mock_sb):
        ids = mgr._get_game_ids(date(2026, 3, 15))
    assert ids == ['0022500001', '0022500002']

def test_get_game_ids_returns_empty_list_when_no_games():
    mgr, _ = _make_manager()
    mock_sb = MagicMock()
    mock_sb.get_data_frames.return_value = [pd.DataFrame({'GAME_ID': []})]
    with patch('data.postgres.player_game_logs.scoreboardv2.ScoreboardV2', return_value=mock_sb):
        ids = mgr._get_game_ids(date(2026, 7, 4))
    assert ids == []


# --- _get_player_rows ---

def test_get_player_rows_returns_list_of_dicts():
    mgr, _ = _make_manager()
    mock_bs = MagicMock()
    mock_bs.get_data_frames.return_value = [pd.DataFrame([_BOX_ROW])]
    with patch('data.postgres.player_game_logs.boxscoretraditionalv2.BoxScoreTraditionalV2', return_value=mock_bs):
        rows = mgr._get_player_rows('0022500001', date(2026, 3, 15), '2025-26')
    assert len(rows) == 1
    assert rows[0]['player_id'] == 2544
    assert rows[0]['pts'] == 28.0
    assert rows[0]['to'] == 3.0

def test_get_player_rows_skips_null_player_id():
    mgr, _ = _make_manager()
    null_row = {**_BOX_ROW, 'PLAYER_ID': float('nan')}
    mock_bs = MagicMock()
    mock_bs.get_data_frames.return_value = [pd.DataFrame([null_row])]
    with patch('data.postgres.player_game_logs.boxscoretraditionalv2.BoxScoreTraditionalV2', return_value=mock_bs):
        rows = mgr._get_player_rows('0022500001', date(2026, 3, 15), '2025-26')
    assert rows == []

def test_get_player_rows_parses_min_string():
    mgr, _ = _make_manager()
    mock_bs = MagicMock()
    mock_bs.get_data_frames.return_value = [pd.DataFrame([_BOX_ROW])]
    with patch('data.postgres.player_game_logs.boxscoretraditionalv2.BoxScoreTraditionalV2', return_value=mock_bs):
        rows = mgr._get_player_rows('0022500001', date(2026, 3, 15), '2025-26')
    assert rows[0]['min'] == 35.0


# --- _upsert_rows ---

def test_upsert_rows_returns_zero_for_empty():
    mgr, _ = _make_manager()
    assert mgr._upsert_rows([]) == 0

def test_upsert_rows_returns_row_count():
    mgr, _ = _make_manager()
    row = {
        'player_id': 2544, 'game_id': '001', 'game_date': date(2026, 3, 15),
        'season': '2025-26', 'team_abbreviation': 'LAL', 'min': 35.0,
        'pts': 28.0, 'reb': 7.0, 'ast': 9.0, 'stl': 1.0, 'blk': 0.0, 'to': 3.0,
        'fgm': 10.0, 'fga': 20.0, 'fg_pct': 0.5, 'fg3m': 2.0, 'fg3a': 6.0,
        'fg3_pct': 0.333, 'ftm': 6.0, 'fta': 8.0, 'ft_pct': 0.75, 'plus_minus': 5.0,
    }
    assert mgr._upsert_rows([row]) == 1

def test_upsert_rows_sql_has_on_conflict():
    mgr, mock_conn = _make_manager()
    row = {
        'player_id': 2544, 'game_id': '001', 'game_date': date(2026, 3, 15),
        'season': '2025-26', 'team_abbreviation': 'LAL', 'min': 35.0,
        'pts': 28.0, 'reb': 7.0, 'ast': 9.0, 'stl': 1.0, 'blk': 0.0, 'to': 3.0,
        'fgm': 10.0, 'fga': 20.0, 'fg_pct': 0.5, 'fg3m': 2.0, 'fg3a': 6.0,
        'fg3_pct': 0.333, 'ftm': 6.0, 'fta': 8.0, 'ft_pct': 0.75, 'plus_minus': 5.0,
    }
    mgr._upsert_rows([row])
    sql = mock_conn.execute.call_args[0][0].text
    assert 'ON CONFLICT' in sql


# --- fetch_and_upsert_date ---

def test_fetch_and_upsert_date_returns_zero_when_no_games():
    mgr, _ = _make_manager()
    mock_sb = MagicMock()
    mock_sb.get_data_frames.return_value = [pd.DataFrame({'GAME_ID': []})]
    with patch('data.postgres.player_game_logs.scoreboardv2.ScoreboardV2', return_value=mock_sb):
        with patch('data.postgres.player_game_logs.time.sleep'):
            result = mgr.fetch_and_upsert_date(date(2026, 7, 4), '2025-26')
    assert result == 0

def test_fetch_and_upsert_date_sleeps_once_per_game():
    mgr, _ = _make_manager()
    mock_sb = MagicMock()
    mock_sb.get_data_frames.return_value = [pd.DataFrame({'GAME_ID': ['001', '002']})]
    mock_bs = MagicMock()
    mock_bs.get_data_frames.return_value = [pd.DataFrame(columns=list(_BOX_ROW.keys()))]
    with patch('data.postgres.player_game_logs.scoreboardv2.ScoreboardV2', return_value=mock_sb):
        with patch('data.postgres.player_game_logs.boxscoretraditionalv2.BoxScoreTraditionalV2', return_value=mock_bs):
            with patch('data.postgres.player_game_logs.time.sleep') as mock_sleep:
                mgr.fetch_and_upsert_date(date(2026, 3, 15), '2025-26')
    assert mock_sleep.call_count == 2


# --- compute_std_dev ---

def test_compute_std_dev_queries_player_game_logs():
    mgr, mock_conn = _make_manager()
    mock_conn.execute.return_value = iter([])
    mgr.compute_std_dev('2025-26')
    sql = mock_conn.execute.call_args[0][0].text
    assert 'player_game_logs' in sql
    assert ':season' in sql

def test_compute_std_dev_returns_player_dict():
    mgr, mock_conn = _make_manager()
    mock_conn.execute.return_value = iter([
        (2544, 3.5, 2.1, 1.8, 0.5, 0.3, 0.9, 0.05, 0.08, 0.06)
    ])
    result = mgr.compute_std_dev('2025-26')
    assert 2544 in result
    assert result[2544]['std_dev_pts'] == 3.5
    assert result[2544]['std_dev_fgp'] == 0.05

def test_compute_std_dev_returns_empty_dict_when_no_data():
    mgr, mock_conn = _make_manager()
    mock_conn.execute.return_value = iter([])
    assert mgr.compute_std_dev('2025-26') == {}
