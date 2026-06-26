from unittest.mock import MagicMock, patch


def _make_mas_patches():
    return [
        patch("agents.multi_agent.multi_agent_system.YahooFantasySportsQuery"),
        patch("agents.multi_agent.multi_agent_system.RAGManager"),
        patch("agents.multi_agent.multi_agent_system.UserContextGatherer"),
        patch("agents.multi_agent.multi_agent_system.YahooFantasyAgent"),
        patch("agents.multi_agent.multi_agent_system.StatisticAgent"),
        patch("agents.multi_agent.multi_agent_system.NewsAgent"),
        patch("agents.multi_agent.multi_agent_system.FantasyAnalystAgent"),
        patch("agents.multi_agent.multi_agent_system.SupervisorAgent"),
        patch("agents.multi_agent.multi_agent_system.get_default_table_name", return_value="test_table"),
    ]


def test_postgres_connection_passed_to_yahoo_agent():
    patches = _make_mas_patches()
    mocks = [p.start() for p in patches]
    try:
        mock_gatherer = mocks[2].return_value
        mock_gatherer.gather.return_value = ""
        mock_gatherer.get_display_info.return_value = {}
        MockYahooAgent = mocks[3]
        mock_connection = MagicMock()
        import agents.multi_agent.multi_agent_system as mas_module
        MultiAgentSystem = mas_module.MultiAgentSystem
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            mas = MultiAgentSystem(disable_rag=True, connection=mock_connection)
        call_kwargs = MockYahooAgent.call_args.kwargs
        assert call_kwargs.get("connection") is mock_connection
    finally:
        for p in patches:
            p.stop()


def test_postgres_connection_created_from_database_url_when_not_passed():
    patches = _make_mas_patches()
    mocks = [p.start() for p in patches]
    try:
        mock_gatherer = mocks[2].return_value
        mock_gatherer.gather.return_value = ""
        mock_gatherer.get_display_info.return_value = {}
        MockYahooAgent = mocks[3]
        with patch("agents.multi_agent.multi_agent_system.PostgresConnection") as MockPG, \
             patch.dict("os.environ", {"OPENAI_API_KEY": "test-key", "DATABASE_URL": "postgresql://test"}):
            mock_pg_instance = MagicMock()
            MockPG.return_value = mock_pg_instance
            import agents.multi_agent.multi_agent_system as mas_module
            MultiAgentSystem = mas_module.MultiAgentSystem
            mas = MultiAgentSystem(disable_rag=True)
            call_kwargs = MockYahooAgent.call_args.kwargs
            assert call_kwargs.get("connection") is mock_pg_instance
    finally:
        for p in patches:
            p.stop()


def test_connection_none_when_no_database_url():
    patches = _make_mas_patches()
    mocks = [p.start() for p in patches]
    try:
        mock_gatherer = mocks[2].return_value
        mock_gatherer.gather.return_value = ""
        mock_gatherer.get_display_info.return_value = {}
        MockYahooAgent = mocks[3]
        import os
        env = {k: v for k, v in os.environ.items() if k not in ("DATABASE_URL", "REDIS_URL")}
        env["OPENAI_API_KEY"] = "test-key"
        with patch.dict("os.environ", env, clear=True):
            import agents.multi_agent.multi_agent_system as mas_module
            MultiAgentSystem = mas_module.MultiAgentSystem
            mas = MultiAgentSystem(disable_rag=True)
        call_kwargs = MockYahooAgent.call_args.kwargs
        assert call_kwargs.get("connection") is None
    finally:
        for p in patches:
            p.stop()
