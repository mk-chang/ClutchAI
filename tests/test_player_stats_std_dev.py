from unittest.mock import MagicMock, patch
from data.postgres.player_stats import PlayerStatsManager, migrate_std_dev_cols, _STD_DEV_COLS


def _make_manager():
    mock_conn = MagicMock()
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_pg = MagicMock()
    mock_pg.get_engine.return_value = mock_engine
    return PlayerStatsManager(mock_pg), mock_conn, mock_pg


def test_std_dev_cols_has_nine_entries():
    assert len(_STD_DEV_COLS) == 9

def test_migrate_std_dev_cols_returns_true():
    _, _, mock_pg = _make_manager()
    assert migrate_std_dev_cols(mock_pg) is True

def test_migrate_std_dev_cols_issues_alter_for_pg_table():
    _, mock_conn, mock_pg = _make_manager()
    migrate_std_dev_cols(mock_pg)
    sqls = [call[0][0].text for call in mock_conn.execute.call_args_list]
    assert any('bball_monsters_player_stats_pg' in s for s in sqls)

def test_migrate_std_dev_cols_issues_alter_for_total_table():
    _, mock_conn, mock_pg = _make_manager()
    migrate_std_dev_cols(mock_pg)
    sqls = [call[0][0].text for call in mock_conn.execute.call_args_list]
    assert any('bball_monsters_player_stats_total' in s for s in sqls)

def test_migrate_std_dev_cols_issues_alter_for_p36_table():
    _, mock_conn, mock_pg = _make_manager()
    migrate_std_dev_cols(mock_pg)
    sqls = [call[0][0].text for call in mock_conn.execute.call_args_list]
    assert any('bball_monsters_player_stats_p36' in s for s in sqls)

def test_migrate_std_dev_cols_includes_all_std_dev_cols():
    _, mock_conn, mock_pg = _make_manager()
    migrate_std_dev_cols(mock_pg)
    combined = ' '.join(call[0][0].text for call in mock_conn.execute.call_args_list)
    for col in _STD_DEV_COLS:
        assert col in combined

def test_migrate_std_dev_cols_returns_false_on_error():
    mock_pg = MagicMock()
    mock_pg.get_engine.side_effect = Exception("DB down")
    assert migrate_std_dev_cols(mock_pg) is False

def test_update_std_devs_returns_zero_for_empty():
    mgr, _, _ = _make_manager()
    assert mgr.update_std_devs({}, '2025-26') == 0

def test_update_std_devs_returns_player_count():
    mgr, _, _ = _make_manager()
    devs = {
        2544:   {c: 1.5 for c in _STD_DEV_COLS},
        203999: {c: 2.0 for c in _STD_DEV_COLS},
    }
    assert mgr.update_std_devs(devs, '2025-26') == 2

def test_update_std_devs_executes_update_for_pg_table():
    mgr, mock_conn, _ = _make_manager()
    mgr.update_std_devs({2544: {c: 1.5 for c in _STD_DEV_COLS}}, '2025-26')
    sqls = [call[0][0].text for call in mock_conn.execute.call_args_list]
    assert any('bball_monsters_player_stats_pg' in s for s in sqls)

def test_update_std_devs_executes_update_for_total_table():
    mgr, mock_conn, _ = _make_manager()
    mgr.update_std_devs({2544: {c: 1.5 for c in _STD_DEV_COLS}}, '2025-26')
    sqls = [call[0][0].text for call in mock_conn.execute.call_args_list]
    assert any('bball_monsters_player_stats_total' in s for s in sqls)

def test_update_std_devs_executes_update_for_p36_table():
    mgr, mock_conn, _ = _make_manager()
    mgr.update_std_devs({2544: {c: 1.5 for c in _STD_DEV_COLS}}, '2025-26')
    sqls = [call[0][0].text for call in mock_conn.execute.call_args_list]
    assert any('bball_monsters_player_stats_p36' in s for s in sqls)

def test_update_std_devs_sql_uses_update_statement():
    mgr, mock_conn, _ = _make_manager()
    mgr.update_std_devs({2544: {c: 1.5 for c in _STD_DEV_COLS}}, '2025-26')
    sqls = [call[0][0].text for call in mock_conn.execute.call_args_list]
    assert all('UPDATE' in s for s in sqls)

def test_update_std_devs_commits():
    mgr, mock_conn, _ = _make_manager()
    mgr.update_std_devs({2544: {c: 1.5 for c in _STD_DEV_COLS}}, '2025-26')
    mock_conn.commit.assert_called_once()

def test_create_tables_sql_includes_std_dev_cols():
    mgr, mock_conn, _ = _make_manager()
    mgr.create_tables()
    combined = ' '.join(call[0][0].text for call in mock_conn.execute.call_args_list)
    for col in _STD_DEV_COLS:
        assert col in combined
