from unittest.mock import MagicMock, patch


def test_yahoo_fantasy_agent_includes_waiver_wire_tools():
    """WaiverWireTool should be instantiated with query."""
    with patch("agents.multi_agent.yahoo_fantasy_agent.WaiverWireTool") as MockWaiver, \
         patch("agents.multi_agent.yahoo_fantasy_agent.YahooFantasyTool") as MockYahoo, \
         patch("agents.multi_agent.base_agent.BasicTool") as MockBasic, \
         patch("agents.multi_agent.base_agent.ChatOpenAI"), \
         patch("agents.multi_agent.base_agent.create_agent"), \
         patch("agents.multi_agent.base_agent.yaml.safe_load", return_value={}):

        mock_ww = MagicMock()
        mock_ww.get_all_tools.return_value = [MagicMock()]
        MockWaiver.return_value = mock_ww

        mock_yahoo = MagicMock()
        mock_yahoo.get_all_tools.return_value = []
        MockYahoo.return_value = mock_yahoo

        mock_basic = MagicMock()
        mock_basic.get_all_tools.return_value = []
        MockBasic.return_value = mock_basic

        mock_query = MagicMock()

        from agents.multi_agent.yahoo_fantasy_agent import YahooFantasyAgent
        agent = YahooFantasyAgent(query=mock_query, openai_api_key="test-key")

    MockWaiver.assert_called_once_with(query=mock_query, debug=False)
    mock_ww.get_all_tools.assert_called_once()
