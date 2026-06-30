import pandas as pd
from unittest.mock import MagicMock, patch

from data.postgres.opponent_defense import OpponentDefenseManager, _compute_ranks, _get_team_abbreviation_map


# --- _compute_ranks ---

_T1 = {
    'TEAM_ID': 1610612738,
    'OPP_PTS': 105.0, 'OPP_REB': 42.0, 'OPP_AST': 24.0,
    'OPP_STL': 7.0,   'OPP_BLK': 4.0,  'OPP_TOV': 14.0,
    'OPP_FG_PCT': 0.44, 'OPP_FG3_PCT': 0.33,
}
_T2 = {
    'TEAM_ID': 1610612737,
    'OPP_PTS': 115.0, 'OPP_REB': 46.0, 'OPP_AST': 28.0,
    'OPP_STL': 8.0,   'OPP_BLK': 5.0,  'OPP_TOV': 12.0,
    'OPP_FG_PCT': 0.48, 'OPP_FG3_PCT': 0.37,
}

def test_compute_ranks_adds_all_rank_columns():
    df = pd.DataFrame([_T1, _T2])
    result = _compute_ranks(df)
    for col in ('rank_pts', 'rank_reb', 'rank_ast', 'rank_stl', 'rank_blk', 'rank_to', 'rank_fg_pct', 'rank_3p_pct'):
        assert col in result.columns

def test_compute_ranks_pts_lower_allowed_is_rank1():
    df = pd.DataFrame([_T1, _T2])
    result = _compute_ranks(df)
    # BOS (1610612738) allows 105 pts (less than ATL's 115), so BOS = rank 1 (best defense)
    bos_rank = result[result['TEAM_ID'] == 1610612738].iloc[0]['rank_pts']
    assert bos_rank == 1

def test_compute_ranks_to_more_forced_is_rank1():
    df = pd.DataFrame([_T1, _T2])
    result = _compute_ranks(df)
    # BOS (1610612738) forces 14 TOV (more than ATL's 12), so BOS = rank 1 for rank_to
    bos_rank = result[result['TEAM_ID'] == 1610612738].iloc[0]['rank_to']
    assert bos_rank == 1


# --- _get_team_abbreviation_map ---

def test_get_team_abbreviation_map_returns_dict_with_int_keys_and_str_values():
    mock_teams = [
        {'id': 1610612738, 'abbreviation': 'BOS'},
        {'id': 1610612737, 'abbreviation': 'ATL'},
    ]
    with patch('data.postgres.opponent_defense.nba_teams.get_teams', return_value=mock_teams):
        result = _get_team_abbreviation_map()
    assert isinstance(result, dict)
    assert result == {1610612738: 'BOS', 1610612737: 'ATL'}
    for team_id, abbr in result.items():
        assert isinstance(team_id, int)
        assert isinstance(abbr, str)


# --- helpers ---

def _make_manager():
    mock_conn = MagicMock()
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_pg = MagicMock()
    mock_pg.get_engine.return_value = mock_engine
    return OpponentDefenseManager(mock_pg), mock_conn


# --- create_table ---

def test_create_table_returns_true():
    mgr, _ = _make_manager()
    assert mgr.create_table() is True

def test_create_table_sql_has_opponent_defense_rankings():
    mgr, mock_conn = _make_manager()
    mgr.create_table()
    sql = mock_conn.execute.call_args[0][0].text
    assert 'opponent_defense_rankings' in sql

def test_create_table_returns_false_on_error():
    mock_pg = MagicMock()
    mock_pg.get_engine.side_effect = Exception("DB down")
    assert OpponentDefenseManager(mock_pg).create_table() is False


# --- fetch_and_upsert ---

def test_fetch_and_upsert_calls_league_dash_team_stats():
    mgr, _ = _make_manager()
    mock_ep = MagicMock()
    mock_ep.get_data_frames.return_value = [pd.DataFrame([_T1, _T2])]
    with patch('data.postgres.opponent_defense.leaguedashteamstats.LeagueDashTeamStats', return_value=mock_ep) as mock_cls:
        with patch('data.postgres.opponent_defense._get_team_abbreviation_map', return_value={1610612738: 'BOS', 1610612737: 'ATL'}):
            mgr.fetch_and_upsert('2025-26')
    mock_cls.assert_called_once()

def test_fetch_and_upsert_returns_row_count():
    mgr, _ = _make_manager()
    mock_ep = MagicMock()
    mock_ep.get_data_frames.return_value = [pd.DataFrame([_T1, _T2])]
    with patch('data.postgres.opponent_defense.leaguedashteamstats.LeagueDashTeamStats', return_value=mock_ep):
        with patch('data.postgres.opponent_defense._get_team_abbreviation_map', return_value={1610612738: 'BOS', 1610612737: 'ATL'}):
            assert mgr.fetch_and_upsert('2025-26') == 2

def test_fetch_and_upsert_sql_has_on_conflict():
    mgr, mock_conn = _make_manager()
    mock_ep = MagicMock()
    mock_ep.get_data_frames.return_value = [pd.DataFrame([_T1])]
    with patch('data.postgres.opponent_defense.leaguedashteamstats.LeagueDashTeamStats', return_value=mock_ep):
        with patch('data.postgres.opponent_defense._get_team_abbreviation_map', return_value={1610612738: 'BOS'}):
            mgr.fetch_and_upsert('2025-26')
    sql = mock_conn.execute.call_args[0][0].text
    assert 'ON CONFLICT' in sql
