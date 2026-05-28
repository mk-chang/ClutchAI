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


def _make_mock_query(players_batch):
    """Return a mock YahooFantasySportsQuery that returns players_batch then raises."""
    query = MagicMock()
    query.get_league_key.return_value = "466.l.58930"

    call_count = [0]

    def fake_query(url, keys):
        call_count[0] += 1
        if call_count[0] == 1:
            return players_batch
        raise Exception("No more players")  # triggers pagination stop

    query.query.side_effect = fake_query
    return query


class TestWaiverWireTool:

    def test_fetch_free_agents_returns_serialized_list(self):
        from agents.tools.waiver_wire import WaiverWireTool

        players = [
            _make_mock_player("Nikola Jokic", "C", "DEN", 99, "team"),
            _make_mock_player("Josh Hart", "SF", "NYK", 45, "freeagents"),
        ]
        query = _make_mock_query(players)
        tool = WaiverWireTool(query=query, redis_client=None)

        result = tool._fetch_free_agents(limit=25)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["name"] == "Nikola Jokic"
        assert result[1]["percent_owned"] == 45

    def test_get_waiver_wire_players_uses_redis_cache(self):
        from agents.tools.waiver_wire import WaiverWireTool

        cached_data = json.dumps([{"name": "Cached Player", "position": "PG", "team": "LAL", "percent_owned": 20, "ownership_type": "freeagents"}])
        redis_client = MagicMock()
        redis_client.get.return_value = cached_data

        query = MagicMock()
        query.get_league_key.return_value = "466.l.58930"
        tool = WaiverWireTool(query=query, redis_client=redis_client)

        result = tool._get_waiver_wire_players(limit=50)

        redis_client.get.assert_called_once_with("clutchai:waiver_wire:466.l.58930")
        query.query.assert_not_called()
        assert "Cached Player" in result

    def test_get_waiver_wire_players_populates_cache_on_miss(self):
        from agents.tools.waiver_wire import WaiverWireTool

        redis_client = MagicMock()
        redis_client.get.return_value = None  # cache miss

        players = [_make_mock_player("Devin Booker", "SG", "PHX", 88, "freeagents")]
        query = _make_mock_query(players)
        tool = WaiverWireTool(query=query, redis_client=redis_client)

        result = tool._get_waiver_wire_players(limit=50)

        redis_client.setex.assert_called_once()
        call_args = redis_client.setex.call_args
        assert call_args[0][0] == "clutchai:waiver_wire:466.l.58930"
        assert call_args[0][1] == 3600
        assert "Devin Booker" in result

    def test_get_waiver_wire_players_no_redis_always_fetches(self):
        from agents.tools.waiver_wire import WaiverWireTool

        players = [_make_mock_player("Jaylen Brown", "SF", "BOS", 75, "freeagents")]
        query = _make_mock_query(players)
        tool = WaiverWireTool(query=query, redis_client=None)

        result = tool._get_waiver_wire_players(limit=50)

        assert "Jaylen Brown" in result

    def test_get_all_tools_returns_langchain_tools(self):
        from agents.tools.waiver_wire import WaiverWireTool

        query = MagicMock()
        query.get_league_key.return_value = "466.l.58930"
        tool = WaiverWireTool(query=query, redis_client=None)

        tools = tool.get_all_tools()
        tool_names = [t.name for t in tools]

        assert "get_waiver_wire_players" in tool_names
        assert "refresh_waiver_wire_cache" in tool_names
