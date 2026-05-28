from unittest.mock import MagicMock, patch


def _make_tool(query_return=None):
    if query_return is None:
        query_return = [{'player_name': 'LeBron James', 'pts': 25.7,
                         'rv': 4.2, 'pv': 12.1, 'three_v': 1.3}]
    mock_conn = MagicMock()
    with patch('agents.tools.player_stats.PostgresConnection', return_value=mock_conn):
        from agents.tools.player_stats import PlayerStatsTool
        tool_obj = PlayerStatsTool(season='2025-26')
    tool_obj._query = MagicMock(return_value=query_return)
    return tool_obj


def test_tool_name_is_basketball_monster_stats():
    tool_obj = _make_tool()
    assert tool_obj.get_all_tools()[0].name == 'basketball_monster_stats'


def test_result_contains_player_and_values():
    tool_obj = _make_tool()
    result = tool_obj.get_all_tools()[0].invoke({'player_name': 'LeBron'})
    assert 'LeBron James' in result
    assert 'rv' in result
    assert 'pv' in result


def test_no_results_returns_message():
    tool_obj = _make_tool(query_return=[])
    result = tool_obj.get_all_tools()[0].invoke({'player_name': 'Nobody'})
    assert 'No players found' in result


def test_stat_type_defaults_to_pg():
    tool_obj = _make_tool()
    tool_obj.get_all_tools()[0].invoke({})
    call_args = tool_obj._query.call_args[1]
    assert call_args['stat_type'] == 'pg'


def test_stat_type_p36_is_passed():
    tool_obj = _make_tool()
    tool_obj.get_all_tools()[0].invoke({'stat_type': 'p36'})
    assert tool_obj._query.call_args[1]['stat_type'] == 'p36'


def test_season_override_is_passed():
    tool_obj = _make_tool()
    tool_obj.get_all_tools()[0].invoke({'season_override': '2023-24'})
    assert tool_obj._query.call_args[1]['season'] == '2023-24'
