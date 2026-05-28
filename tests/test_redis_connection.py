from unittest.mock import patch, MagicMock
import pytest


def test_redis_connection_from_env():
    with patch("redis.Redis.from_url") as mock_from_url:
        mock_client = MagicMock()
        mock_from_url.return_value = mock_client

        from data.redis.connection import RedisConnection
        conn = RedisConnection(redis_url="redis://localhost:6379")

        mock_from_url.assert_called_once_with("redis://localhost:6379", decode_responses=True)
        assert conn.get_client() is mock_client


def test_redis_connection_from_env_var():
    with patch.dict("os.environ", {"REDIS_URL": "redis://env-host:6379"}):
        with patch("redis.Redis.from_url") as mock_from_url:
            mock_from_url.return_value = MagicMock()

            from data.redis.connection import RedisConnection
            import importlib
            import data.redis.connection
            importlib.reload(data.redis.connection)
            conn = data.redis.connection.RedisConnection()

            mock_from_url.assert_called_once_with("redis://env-host:6379", decode_responses=True)


def test_redis_connection_raises_without_url():
    with patch.dict("os.environ", {}, clear=True):
        from data.redis.connection import RedisConnection
        with pytest.raises(ValueError, match="REDIS_URL"):
            RedisConnection()
