import json
from typing import List, Optional
from langchain_core.tools import tool

from .base import ClutchAITool
from logger import get_logger

logger = get_logger(__name__)

_CACHE_TTL = 3600  # 1 hour
_BATCH_SIZE = 25


class WaiverWireTool(ClutchAITool):
    """
    Fetches free agents from Yahoo Fantasy API and caches results in Redis.

    Cache key: clutchai:waiver_wire:{league_key}
    TTL: 3600 seconds (1 hour)

    Uses yfpy's query() method directly to add status=FA and ownership data
    to the standard league players URL.
    """

    def __init__(self, query, redis_client=None, debug: bool = False):
        """
        Args:
            query: YahooFantasySportsQuery instance
            redis_client: redis.Redis client (optional; disables caching if None)
            debug: Enable debug logging
        """
        super().__init__(debug=debug)
        self.query = query
        self.redis_client = redis_client

    def _cache_key(self) -> str:
        return f"clutchai:waiver_wire:{self.query.get_league_key()}"

    def _serialize_player(self, player) -> dict:
        try:
            return {
                "name": player.name.full,
                "position": player.primary_position,
                "team": player.editorial_team_abbr,
                "percent_owned": player.percent_owned.value,
                "ownership_type": player.ownership.ownership_type,
            }
        except Exception as e:
            logger.debug(f"Error serializing player: {e}")
            return {"name": str(player), "position": "", "team": "", "percent_owned": 0, "ownership_type": ""}

    def _fetch_free_agents(self, limit: int = 50) -> List[dict]:
        """Fetch free agents from Yahoo API with pagination."""
        league_key = self.query.get_league_key()
        players = []
        start = 0

        while len(players) < limit:
            count = min(_BATCH_SIZE, limit - len(players))
            url = (
                f"https://fantasysports.yahooapis.com/fantasy/v2/league/{league_key}/players;"
                f"status=FA;start={start};count={count};out=ownership,percent_owned"
            )
            try:
                batch = self.query.query(url, ["league", "players"])
                if not batch:
                    break
                if not isinstance(batch, list):
                    batch = [batch]
                players.extend(self._serialize_player(p) for p in batch)
                if len(batch) < count:
                    break
                start += count
            except Exception as e:
                logger.debug(f"No more free agents at start={start}: {e}")
                break

        logger.info(f"Fetched {len(players)} free agents from Yahoo API")
        return players

    def _get_waiver_wire_players(self, limit: int = 50) -> str:
        """Return free agents as JSON string, using Redis cache when available."""
        if self.redis_client is not None:
            cached = self.redis_client.get(self._cache_key())
            if cached:
                logger.debug("Waiver wire cache hit")
                return cached

        players = self._fetch_free_agents(limit=limit)
        result = json.dumps(players, indent=2)

        if self.redis_client is not None:
            self.redis_client.setex(self._cache_key(), _CACHE_TTL, result)
            logger.debug(f"Cached {len(players)} free agents (TTL={_CACHE_TTL}s)")

        return result

    def get_all_tools(self) -> list:
        get_waiver_wire_players = self._get_waiver_wire_players
        cache_key = self._cache_key
        redis_client = self.redis_client
        fetch_free_agents = self._fetch_free_agents

        @tool
        def get_waiver_wire_players(limit: int = 50) -> str:
            """
            Get available free agents (waiver wire players) in the fantasy league.

            Returns a JSON list of free agents with name, position, NBA team,
            percent_owned, and ownership_type. Results are cached for 1 hour.

            Args:
                limit: Maximum number of players to return (default 50)

            Returns:
                JSON string with list of free agent player objects
            """
            return get_waiver_wire_players(limit=limit)

        @tool
        def refresh_waiver_wire_cache() -> str:
            """
            Force a fresh fetch of waiver wire players from Yahoo, bypassing the Redis cache.

            Use this when you need up-to-date data after recent roster moves.

            Returns:
                JSON string with updated list of free agent player objects
            """
            if redis_client is not None:
                redis_client.delete(cache_key())
                logger.info("Waiver wire cache cleared")
            players = fetch_free_agents(limit=50)
            result = json.dumps(players, indent=2)
            if redis_client is not None:
                redis_client.setex(cache_key(), _CACHE_TTL, result)
            return result

        return [get_waiver_wire_players, refresh_waiver_wire_cache]
