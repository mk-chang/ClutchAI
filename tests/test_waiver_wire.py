import json
from unittest.mock import MagicMock, patch
import pytest


def _make_mock_player(name, position, team, percent_owned, ownership_type="freeagents"):
    player = MagicMock()
    player.name.full = name
    player.primary_position = position
    player.editorial_team_abbr = team
    player.percent_owned.value = percent_owned
    player.ownership.ownership_type = ownership_type
    return player


def _make_mock_query(players_batch, tx_id=10):
    query = MagicMock()
    query.get_league_key.return_value = "466.l.58930"

    call_count = [0]

    def fake_query(url, keys):
        call_count[0] += 1
        if call_count[0] == 1:
            return players_batch
        raise Exception("No more players")

    query.query.side_effect = fake_query

    mock_tx = MagicMock()
    mock_tx.transaction_id = tx_id
    query.get_league_transactions.return_value = [mock_tx]

    return query


class TestWaiverWireTool:

    def test_fetch_free_agents_returns_serialized_list(self):
        from agents.tools.waiver_wire import WaiverWireTool

        players = [
            _make_mock_player("Josh Hart", "SF", "NYK", 45),
            _make_mock_player("Devin Booker", "SG", "PHX", 88),
        ]
        query = _make_mock_query(players)
        tool = WaiverWireTool(query=query, connection=None)

        result = tool._fetch_free_agents(limit=50)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["name"] == "Josh Hart"
        assert result[1]["percent_owned"] == 88

    def test_get_latest_tx_id_returns_max_id(self):
        from agents.tools.waiver_wire import WaiverWireTool

        query = MagicMock()
        query.get_league_key.return_value = "466.l.58930"
        tx1, tx2 = MagicMock(), MagicMock()
        tx1.transaction_id = 5
        tx2.transaction_id = 12
        query.get_league_transactions.return_value = [tx1, tx2]

        tool = WaiverWireTool(query=query, connection=None)
        assert tool._get_latest_tx_id() == 12

    def test_get_latest_tx_id_returns_none_on_failure(self):
        from agents.tools.waiver_wire import WaiverWireTool

        query = MagicMock()
        query.get_league_key.return_value = "466.l.58930"
        query.get_league_transactions.side_effect = Exception("API error")

        tool = WaiverWireTool(query=query, connection=None)
        assert tool._get_latest_tx_id() is None

    def test_get_waiver_wire_players_uses_store_on_cache_hit(self):
        from agents.tools.waiver_wire import WaiverWireTool

        cached_players = [{"name": "Cached Player", "position": "PG", "team": "LAL",
                           "percent_owned": 20, "ownership_type": "freeagents"}]
        mock_store = MagicMock()
        mock_store.get.return_value = {"players": cached_players, "last_tx_id": 10}

        query = _make_mock_query([], tx_id=10)

        with patch("agents.tools.waiver_wire.WaiverWireStore", return_value=mock_store):
            tool = WaiverWireTool(query=query, connection=MagicMock())
            result = tool._get_waiver_wire_players()

        mock_store.get.assert_called_once_with("466.l.58930")
        query.query.assert_not_called()
        assert "Cached Player" in result

    def test_get_waiver_wire_players_refetches_on_new_transaction(self):
        from agents.tools.waiver_wire import WaiverWireTool

        cached_players = [{"name": "Old Player", "position": "PG", "team": "LAL",
                           "percent_owned": 20, "ownership_type": "freeagents"}]
        mock_store = MagicMock()
        mock_store.get.return_value = {"players": cached_players, "last_tx_id": 9}

        new_players = [_make_mock_player("New Player", "SG", "BOS", 55)]
        query = _make_mock_query(new_players, tx_id=10)

        with patch("agents.tools.waiver_wire.WaiverWireStore", return_value=mock_store):
            tool = WaiverWireTool(query=query, connection=MagicMock())
            result = tool._get_waiver_wire_players()

        query.query.assert_called()
        assert mock_store.put.call_args[0][0] == "466.l.58930"
        assert mock_store.put.call_args[0][2] == 10
        assert "New Player" in result

    def test_get_waiver_wire_players_no_connection_always_fetches(self):
        from agents.tools.waiver_wire import WaiverWireTool

        players = [_make_mock_player("Jaylen Brown", "SF", "BOS", 75)]
        query = _make_mock_query(players)
        tool = WaiverWireTool(query=query, connection=None)

        result = tool._get_waiver_wire_players()

        assert "Jaylen Brown" in result

    def test_get_all_tools_returns_langchain_tools(self):
        from agents.tools.waiver_wire import WaiverWireTool

        query = MagicMock()
        query.get_league_key.return_value = "466.l.58930"
        tool = WaiverWireTool(query=query, connection=None)

        tools = tool.get_all_tools()
        tool_names = [t.name for t in tools]

        assert "get_waiver_wire_players" in tool_names
        assert "refresh_waiver_wire_cache" in tool_names
