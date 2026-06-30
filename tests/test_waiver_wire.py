import json
from unittest.mock import MagicMock


def _make_mock_player(name, position, team, percent_owned, ownership_type="freeagents"):
    player = MagicMock()
    player.name.full = name
    player.primary_position = position
    player.editorial_team_abbr = team
    player.percent_owned.value = percent_owned
    player.ownership.ownership_type = ownership_type
    return player


def _make_mock_query(players_batch):
    query = MagicMock()
    query.get_league_key.return_value = "466.l.58930"

    call_count = [0]

    def fake_query(url, keys):
        call_count[0] += 1
        if call_count[0] == 1:
            return players_batch
        raise Exception("No more players")

    query.query.side_effect = fake_query
    return query


class TestWaiverWireTool:

    def test_fetch_free_agents_returns_serialized_list(self):
        from agents.tools.waiver_wire import WaiverWireTool

        players = [
            _make_mock_player("Josh Hart", "SF", "NYK", 45),
            _make_mock_player("Devin Booker", "SG", "PHX", 88),
        ]
        query = _make_mock_query(players)
        tool = WaiverWireTool(query=query)

        result = tool._fetch_free_agents(limit=50)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["name"] == "Josh Hart"
        assert result[1]["percent_owned"] == 88

    def test_fetch_free_agents_handles_serialization_error(self):
        from agents.tools.waiver_wire import WaiverWireTool

        bad_player = MagicMock()
        bad_player.name.full  # raises AttributeError
        type(bad_player).name = MagicMock(side_effect=AttributeError)

        query = _make_mock_query([bad_player])
        tool = WaiverWireTool(query=query)

        result = tool._fetch_free_agents(limit=50)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["name"] != ""  # fallback string representation

    def test_get_all_tools_returns_get_waiver_wire_players(self):
        from agents.tools.waiver_wire import WaiverWireTool

        query = MagicMock()
        query.get_league_key.return_value = "466.l.58930"
        tool = WaiverWireTool(query=query)

        tools = tool.get_all_tools()
        tool_names = [t.name for t in tools]

        assert "get_waiver_wire_players" in tool_names
        assert len(tools) == 1

    def test_get_waiver_wire_players_returns_json(self):
        from agents.tools.waiver_wire import WaiverWireTool

        players = [_make_mock_player("Jaylen Brown", "SF", "BOS", 75)]
        query = _make_mock_query(players)
        tool = WaiverWireTool(query=query)

        tools = tool.get_all_tools()
        result = tools[0].invoke({})

        data = json.loads(result)
        assert isinstance(data, list)
        assert data[0]["name"] == "Jaylen Brown"
