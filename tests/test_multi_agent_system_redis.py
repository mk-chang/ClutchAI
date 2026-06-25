from unittest.mock import MagicMock, patch

import agents.multi_agent.multi_agent_system as mas_module


def _make_mas_patches():
    """Return patches for all heavy MultiAgentSystem dependencies."""
    return [
        patch.object(mas_module, "YahooFantasySportsQuery"),
        patch.object(mas_module, "RAGManager"),
        patch.object(mas_module, "UserContextGatherer"),
        patch.object(mas_module, "YahooFantasyAgent"),
        patch.object(mas_module, "StatisticAgent"),
        patch.object(mas_module, "NewsAgent"),
        patch.object(mas_module, "FantasyAnalystAgent"),
        patch.object(mas_module, "SupervisorAgent"),
    ]


def test_redis_client_passed_to_yahoo_agent_when_url_provided():
    """When REDIS_URL is set, MultiAgentSystem builds a redis client and passes it to YahooFantasyAgent."""
    patches = _make_mas_patches()
    mocks = [p.start() for p in patches]

    try:
        mock_gatherer = mocks[2].return_value
        mock_gatherer.gather.return_value = ""
        mock_gatherer.get_display_info.return_value = {}

        MockYahooAgent = mocks[3]

        mock_redis_conn = MagicMock()
        mock_redis_client = MagicMock()
        mock_redis_conn.get_client.return_value = mock_redis_client

        with patch.object(mas_module, "RedisConnection", return_value=mock_redis_conn) as MockRedis, \
             patch.dict("os.environ", {"REDIS_URL": "redis://localhost:6379",
                                       "OPENAI_API_KEY": "test-key",
                                       "DISABLE_RAG": "true",
                                       "VECTOR_TABLE": "test_table"}):

            mas = mas_module.MultiAgentSystem(disable_rag=True)

            MockRedis.assert_called_once_with(redis_url="redis://localhost:6379")
            call_kwargs = MockYahooAgent.call_args.kwargs
            assert call_kwargs.get("redis_client") is mock_redis_client
    finally:
        for p in patches:
            p.stop()


def test_redis_client_none_when_no_url():
    """When no REDIS_URL, YahooFantasyAgent receives redis_client=None."""
    patches = _make_mas_patches()
    mocks = [p.start() for p in patches]

    try:
        mock_gatherer = mocks[2].return_value
        mock_gatherer.gather.return_value = ""
        mock_gatherer.get_display_info.return_value = {}

        MockYahooAgent = mocks[3]

        import os
        env_overrides = {"OPENAI_API_KEY": "test-key", "DISABLE_RAG": "true", "VECTOR_TABLE": "test_table"}
        with patch.object(mas_module, "RedisConnection") as MockRedis, \
             patch.dict("os.environ", env_overrides):

            os.environ.pop("REDIS_URL", None)

            mas = mas_module.MultiAgentSystem(disable_rag=True)

            MockRedis.assert_not_called()
            call_kwargs = MockYahooAgent.call_args.kwargs
            assert call_kwargs.get("redis_client") is None
    finally:
        for p in patches:
            p.stop()
