import json
from datetime import date
from unittest.mock import MagicMock

from agents.tools.player_stats_db import PlayerStatsDbTool, _is_safe_sql


# --- _is_safe_sql ---

def test_is_safe_sql_allows_select():
    assert _is_safe_sql('SELECT * FROM player_game_logs') is True

def test_is_safe_sql_allows_select_with_leading_whitespace():
    assert _is_safe_sql('  SELECT player_id FROM player_game_logs') is True

def test_is_safe_sql_rejects_insert():
    assert _is_safe_sql('INSERT INTO player_game_logs VALUES (1)') is False

def test_is_safe_sql_rejects_update():
    assert _is_safe_sql('UPDATE player_game_logs SET pts=0') is False

def test_is_safe_sql_rejects_drop():
    assert _is_safe_sql('DROP TABLE player_game_logs') is False

def test_is_safe_sql_rejects_delete():
    assert _is_safe_sql('DELETE FROM player_game_logs') is False


# --- helpers ---

def _make_tools():
    mock_ctx = MagicMock()
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_ctx
    mock_pg = MagicMock()
    mock_pg.get_engine.return_value = mock_engine
    tool_obj = PlayerStatsDbTool(connection=mock_pg)
    tools = {t.name: t for t in tool_obj.get_all_tools()}
    return tools, mock_ctx


# --- get_all_tools ---

def test_get_all_tools_returns_four_tools():
    tools, _ = _make_tools()
    assert len(tools) == 4

def test_get_all_tools_has_expected_names():
    tools, _ = _make_tools()
    assert 'get_recent_form' in tools
    assert 'get_schedule_density' in tools
    assert 'get_season_trends' in tools
    assert 'query_stats_db' in tools


# --- get_recent_form ---

def test_get_recent_form_returns_json_with_recent_and_season_stats():
    tools, mock_ctx = _make_tools()
    keys = [
        'recent_games', 'recent_pts', 'recent_reb', 'recent_ast',
        'recent_stl', 'recent_blk', 'recent_tov', 'recent_fg_pct', 'recent_3p_pct', 'recent_ft_pct',
        'season_pts', 'season_reb', 'season_ast', 'season_stl', 'season_blk', 'season_tov',
        'season_fg_pct', 'season_3p_pct', 'season_ft_pct',
    ]
    mock_ctx.execute.return_value.fetchone.return_value = (10, 25.0, 7.0, 8.5, 1.2, 0.5, 3.0, 0.510, 0.360, 0.740,
                                                            24.0, 7.2, 8.1, 1.1, 0.4, 3.2, 0.500, 0.350, 0.730)
    mock_ctx.execute.return_value.keys.return_value = keys
    result = json.loads(tools['get_recent_form'].invoke({'player_id': 2544}))
    assert result['recent_pts'] == 25.0
    assert result['season_pts'] == 24.0

def test_get_recent_form_returns_error_for_missing_player():
    tools, mock_ctx = _make_tools()
    mock_ctx.execute.return_value.fetchone.return_value = (0,) + (None,) * 18
    mock_ctx.execute.return_value.keys.return_value = ['recent_games'] + ['x'] * 18
    result = json.loads(tools['get_recent_form'].invoke({'player_id': 99999}))
    assert 'error' in result

def test_get_recent_form_default_n_games_is_10():
    tools, mock_ctx = _make_tools()
    mock_ctx.execute.return_value.fetchone.return_value = (0,) + (None,) * 18
    mock_ctx.execute.return_value.keys.return_value = ['recent_games'] + ['x'] * 18
    tools['get_recent_form'].invoke({'player_id': 2544})
    params = mock_ctx.execute.call_args[0][1]
    assert params['n_games'] == 10


# --- get_schedule_density ---

def test_get_schedule_density_returns_game_list():
    tools, mock_ctx = _make_tools()
    mock_ctx.execute.return_value.__iter__ = lambda s: iter([
        (date(2026, 3, 16), 'MIA', 'home'),
        (date(2026, 3, 18), 'BKN', 'away'),
    ])
    result = json.loads(tools['get_schedule_density'].invoke({'team_abbreviation': 'BOS'}))
    assert result['game_count'] == 2
    assert result['games'][0]['opponent'] == 'MIA'

def test_get_schedule_density_default_days_is_7():
    tools, mock_ctx = _make_tools()
    mock_ctx.execute.return_value.__iter__ = lambda s: iter([])
    tools['get_schedule_density'].invoke({'team_abbreviation': 'BOS'})
    params = mock_ctx.execute.call_args[0][1]
    assert params['days'] == 7


# --- get_season_trends ---

def test_get_season_trends_returns_monthly_trends():
    tools, mock_ctx = _make_tools()
    keys = ['month', 'games', 'pts', 'reb', 'ast', 'stl', 'blk', 'tov', 'fg_pct', 'fg3_pct', 'ft_pct']
    mock_ctx.execute.return_value.fetchall.return_value = [
        ('2025-10', 10, 24.0, 7.0, 8.0, 1.1, 0.4, 3.0, 0.50, 0.35, 0.73)
    ]
    mock_ctx.execute.return_value.keys.return_value = keys
    result = json.loads(tools['get_season_trends'].invoke({'player_id': 2544}))
    assert len(result['monthly_trends']) == 1
    assert result['monthly_trends'][0]['month'] == '2025-10'

def test_get_season_trends_returns_error_for_missing_player():
    tools, mock_ctx = _make_tools()
    mock_ctx.execute.return_value.fetchall.return_value = []
    result = json.loads(tools['get_season_trends'].invoke({'player_id': 99999}))
    assert 'error' in result


# --- query_stats_db ---

def test_query_stats_db_executes_select_and_returns_rows():
    tools, mock_ctx = _make_tools()
    mock_ctx.execute.return_value.fetchall.return_value = [(2544, 'LeBron James')]
    mock_ctx.execute.return_value.keys.return_value = ['player_id', 'player_name']
    result = json.loads(tools['query_stats_db'].invoke({'sql': 'SELECT player_id, player_name FROM player_game_logs LIMIT 1'}))
    assert result[0]['player_id'] == 2544

def test_query_stats_db_rejects_drop_without_executing():
    tools, mock_ctx = _make_tools()
    result = json.loads(tools['query_stats_db'].invoke({'sql': 'DROP TABLE player_game_logs'}))
    assert 'error' in result
    mock_ctx.execute.assert_not_called()

def test_query_stats_db_rejects_insert_without_executing():
    tools, mock_ctx = _make_tools()
    result = json.loads(tools['query_stats_db'].invoke({'sql': 'INSERT INTO player_game_logs VALUES (1)'}))
    assert 'error' in result
    mock_ctx.execute.assert_not_called()
