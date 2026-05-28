import pandas as pd
from datetime import date
from unittest.mock import MagicMock, patch

_API_ROW = {
    'PLAYER_ID': 2544, 'PLAYER_NAME': 'LeBron James',
    'TEAM_ABBREVIATION': 'LAL', 'AGE': 40.0, 'GP': 70,
    'MIN': 35.5, 'PTS': 25.7, 'REB': 7.4, 'AST': 8.3,
    'STL': 1.3, 'BLK': 0.5, 'TOV': 3.5,
    'FGM': 9.2, 'FGA': 18.0, 'FG_PCT': 0.511,
    'FG3M': 2.1, 'FG3A': 5.8, 'FG3_PCT': 0.362,
    'FTM': 5.0, 'FTA': 6.8, 'FT_PCT': 0.735,
    'OREB': 1.3, 'DREB': 6.1, 'PF': 1.4, 'PLUS_MINUS': 3.2,
}


def _make_manager():
    mock_conn = MagicMock()
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_pg = MagicMock()
    mock_pg.get_engine.return_value = mock_engine
    from data.postgres.player_stats import PlayerStatsManager
    return PlayerStatsManager(mock_pg), mock_conn


# --- create_tables ---

def test_create_tables_returns_true():
    mgr, _ = _make_manager()
    assert mgr.create_tables() is True

def test_create_tables_creates_pg_table():
    mgr, mock_conn = _make_manager()
    mgr.create_tables()
    sqls = [call[0][0].text for call in mock_conn.execute.call_args_list]
    assert any('bball_monsters_player_stats_pg' in s for s in sqls)

def test_create_tables_creates_total_table():
    mgr, mock_conn = _make_manager()
    mgr.create_tables()
    sqls = [call[0][0].text for call in mock_conn.execute.call_args_list]
    assert any('bball_monsters_player_stats_total' in s for s in sqls)

def test_create_tables_creates_p36_table():
    mgr, mock_conn = _make_manager()
    mgr.create_tables()
    sqls = [call[0][0].text for call in mock_conn.execute.call_args_list]
    assert any('bball_monsters_player_stats_p36' in s for s in sqls)

def test_create_tables_returns_false_on_error():
    mock_pg = MagicMock()
    mock_pg.get_engine.side_effect = Exception("DB down")
    from data.postgres.player_stats import PlayerStatsManager
    assert PlayerStatsManager(mock_pg).create_tables() is False


# --- fetch ---

def test_fetch_makes_three_api_calls():
    endpoints = [MagicMock(), MagicMock(), MagicMock()]
    for ep in endpoints:
        ep.get_data_frames.return_value = [pd.DataFrame([_API_ROW])]
    with patch('data.postgres.player_stats.LeagueDashPlayerStats', side_effect=endpoints):
        from data.postgres.player_stats import PlayerStatsManager
        pg, tot, p36 = PlayerStatsManager(MagicMock()).fetch('2025-26')
    assert len(pg) == len(tot) == len(p36) == 1

def test_fetch_returns_three_dataframes():
    endpoints = [MagicMock(), MagicMock(), MagicMock()]
    for ep in endpoints:
        ep.get_data_frames.return_value = [pd.DataFrame([_API_ROW])]
    with patch('data.postgres.player_stats.LeagueDashPlayerStats', side_effect=endpoints):
        from data.postgres.player_stats import PlayerStatsManager
        result = PlayerStatsManager(MagicMock()).fetch('2025-26')
    assert len(result) == 3


# --- upsert ---

def test_upsert_pg_executes_for_each_row():
    mgr, mock_conn = _make_manager()
    df = pd.DataFrame([_API_ROW])
    mgr.upsert_pg(df, '2025-26')
    assert mock_conn.execute.call_count == 1

def test_upsert_pg_sql_has_on_conflict():
    mgr, mock_conn = _make_manager()
    mgr.upsert_pg(pd.DataFrame([_API_ROW]), '2025-26')
    sql = mock_conn.execute.call_args[0][0].text
    assert 'ON CONFLICT' in sql
    assert 'bball_monsters_player_stats_pg' in sql

def test_upsert_total_sql_targets_total_table():
    mgr, mock_conn = _make_manager()
    mgr.upsert_total(pd.DataFrame([_API_ROW]), '2025-26')
    sql = mock_conn.execute.call_args[0][0].text
    assert 'bball_monsters_player_stats_total' in sql

def test_upsert_p36_sql_targets_p36_table():
    mgr, mock_conn = _make_manager()
    mgr.upsert_p36(pd.DataFrame([_API_ROW]), '2025-26')
    sql = mock_conn.execute.call_args[0][0].text
    assert 'bball_monsters_player_stats_p36' in sql

def test_upsert_maps_player_columns():
    mgr, mock_conn = _make_manager()
    mgr.upsert_pg(pd.DataFrame([_API_ROW]), '2025-26')
    params = mock_conn.execute.call_args[0][1]
    assert params['player_id'] == 2544
    assert params['player_name'] == 'LeBron James'
    assert params['pts'] == 25.7
    assert params['fg3m'] == 2.1

def test_upsert_commits():
    mgr, mock_conn = _make_manager()
    mgr.upsert_pg(pd.DataFrame([_API_ROW, _API_ROW]), '2025-26')
    mock_conn.commit.assert_called_once()


# --- season utilities ---

def test_is_nba_season_true_in_october():
    from data.postgres.player_stats import is_nba_season
    assert is_nba_season(date(2025, 10, 15)) is True

def test_is_nba_season_true_in_june():
    from data.postgres.player_stats import is_nba_season
    assert is_nba_season(date(2026, 6, 10)) is True

def test_is_nba_season_false_in_july():
    from data.postgres.player_stats import is_nba_season
    assert is_nba_season(date(2026, 7, 1)) is False

def test_is_nba_season_false_in_august():
    from data.postgres.player_stats import is_nba_season
    assert is_nba_season(date(2026, 8, 15)) is False

def test_is_nba_season_false_in_september():
    from data.postgres.player_stats import is_nba_season
    assert is_nba_season(date(2026, 9, 30)) is False

def test_current_season_october():
    from data.postgres.player_stats import current_season
    assert current_season(date(2025, 10, 20)) == '2025-26'

def test_current_season_january():
    from data.postgres.player_stats import current_season
    assert current_season(date(2026, 1, 15)) == '2025-26'

def test_last_n_seasons_returns_correct_list():
    from data.postgres.player_stats import last_n_seasons
    assert last_n_seasons(date(2026, 5, 27), n=3) == ['2023-24', '2024-25', '2025-26']
