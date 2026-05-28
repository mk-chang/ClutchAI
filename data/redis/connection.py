import os
import redis


class RedisConnection:

    def __init__(self, redis_url: str = None):
        url = redis_url or os.environ.get("REDIS_URL")
        if not url:
            raise ValueError(
                "REDIS_URL environment variable is required. "
                "Railway injects this automatically when a Redis plugin is attached."
            )
        self._client = redis.Redis.from_url(url, decode_responses=True)

    def get_client(self) -> redis.Redis:
        return self._client
