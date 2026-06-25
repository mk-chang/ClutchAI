from unittest.mock import MagicMock, patch
import json
import pytest


def _make_mock_connection(fetchone_result=None):
    """Build a mock PostgresConnection whose engine returns controlled row data."""
    mock_conn_ctx = MagicMock()
    mock_execute_result = MagicMock()
    mock_execute_result.fetchone.return_value = fetchone_result
    mock_conn_ctx.__enter__ = MagicMock(return_value=mock_conn_ctx)
    mock_conn_ctx.__exit__ = MagicMock(return_value=False)
    mock_conn_ctx.execute.return_value = mock_execute_result
    mock_conn_ctx.commit = MagicMock()

    mock_engine = MagicMock()
    mock_engine.connect.return_value = mock_conn_ctx

    mock_connection = MagicMock()
    mock_connection.get_engine.return_value = mock_engine
    return mock_connection, mock_conn_ctx


class TestWaiverWireStore:

    def test_create_table_executes_ddl(self):
        from data.postgres.waiver_wire import WaiverWireStore

        mock_connection, mock_conn_ctx = _make_mock_connection()
        store = WaiverWireStore(mock_connection)
        store.create_table()

        assert mock_conn_ctx.execute.called
        sql = str(mock_conn_ctx.execute.call_args[0][0])
        assert "waiver_wire_cache" in sql
        mock_conn_ctx.commit.assert_called_once()

    def test_get_returns_none_when_no_row(self):
        from data.postgres.waiver_wire import WaiverWireStore

        mock_connection, _ = _make_mock_connection(fetchone_result=None)
        store = WaiverWireStore(mock_connection)

        result = store.get("466.l.58930")
        assert result is None

    def test_get_returns_dict_when_row_exists(self):
        from data.postgres.waiver_wire import WaiverWireStore

        players_data = [{"name": "Josh Hart", "position": "SF", "team": "NYK",
                         "percent_owned": 45, "ownership_type": "freeagents"}]
        mock_row = MagicMock()
        mock_row.players = players_data
        mock_row.last_tx_id = 42

        mock_connection, _ = _make_mock_connection(fetchone_result=mock_row)
        store = WaiverWireStore(mock_connection)

        result = store.get("466.l.58930")
        assert result == {"players": players_data, "last_tx_id": 42}

    def test_put_executes_upsert(self):
        from data.postgres.waiver_wire import WaiverWireStore

        mock_connection, mock_conn_ctx = _make_mock_connection()
        store = WaiverWireStore(mock_connection)

        players = [{"name": "Devin Booker", "position": "SG", "team": "PHX",
                    "percent_owned": 88, "ownership_type": "freeagents"}]
        store.put("466.l.58930", players, last_tx_id=99)

        assert mock_conn_ctx.execute.called
        sql = str(mock_conn_ctx.execute.call_args[0][0])
        assert "INSERT" in sql
        assert "ON CONFLICT" in sql
        mock_conn_ctx.commit.assert_called_once()

    def test_delete_executes_delete_sql(self):
        from data.postgres.waiver_wire import WaiverWireStore

        mock_connection, mock_conn_ctx = _make_mock_connection()
        store = WaiverWireStore(mock_connection)
        store.delete("466.l.58930")

        assert mock_conn_ctx.execute.called
        sql = str(mock_conn_ctx.execute.call_args[0][0])
        assert "DELETE" in sql
        mock_conn_ctx.commit.assert_called_once()
