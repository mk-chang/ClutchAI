import json
from typing import List, Optional
from langchain_core.tools import tool

from .base import ClutchAITool
from data.postgres.connection import PostgresConnection
from data.postgres.waiver_wire import WaiverWireStore
from logger import get_logger

logger = get_logger(__name__)

_BATCH_SIZE = 25


class WaiverWireTool(ClutchAITool):
    """
    Fetches free agents from Yahoo Fantasy API and persists results in Postgres.

    Cache invalidation: on each query, fetches the latest transaction_id from
    Yahoo and compares it to the stored value. Stale on mismatch; re-fetches.
    Data persists indefinitely — no TTL expiry.
    """

    def __init__(self, query, connection: Optional[PostgresConnection] = None, debug: bool = False):
        """
        Args:
            query: YahooFantasySportsQuery instance
            connection: PostgresConnection for persistent caching (optional)
            debug: Enable debug logging
        """
        super().__init__(debug=debug)
        self.query = query
        self.store = None
        if connection is not None:
            try:
                self.store = WaiverWireStore(connection)
                self.store.create_table()
            except Exception as e:
                logger.warning(f"WaiverWireStore unavailable, caching disabled: {e}")
                self.store = None

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

    def _get_latest_tx_id(self) -> Optional[int]:
        """Return the highest transaction_id in the league, or None on failure."""
        try:
            transactions = self.query.get_league_transactions()
            if not transactions:
                return None
            ids = [t.transaction_id for t in transactions if t.transaction_id is not None]
            return max(ids) if ids else None
        except Exception as e:
            logger.warning(f"Could not fetch transactions for cache check: {e}")
            return None

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

    def _get_waiver_wire_players(self) -> str:
        """
        Return free agents as JSON string.

        Checks Postgres for a cached entry. If the latest transaction_id
        matches the stored last_tx_id, returns cached data. Otherwise
        re-fetches from Yahoo and updates the store.
        """
        latest_tx_id = self._get_latest_tx_id()
        league_key = self.query.get_league_key()

        if self.store is not None:
            try:
                cached = self.store.get(league_key)
                if cached:
                    if latest_tx_id is None or cached["last_tx_id"] == latest_tx_id:
                        logger.debug("Waiver wire cache hit")
                        return json.dumps(cached["players"], indent=2)
            except Exception as e:
                logger.warning(f"Waiver wire store read failed, fetching live: {e}")

        players = self._fetch_free_agents()

        if self.store is not None and latest_tx_id is not None:
            try:
                self.store.put(league_key, players, latest_tx_id)
            except Exception as e:
                logger.warning(f"Waiver wire store write failed: {e}")

        return json.dumps(players, indent=2)

    def get_all_tools(self) -> list:
        _ww_fetch = self._get_waiver_wire_players
        store = self.store
        league_key_fn = self.query.get_league_key
        fetch_free_agents = self._fetch_free_agents
        get_latest_tx_id = self._get_latest_tx_id

        @tool
        def get_waiver_wire_players() -> str:
            """
            Get available free agents (waiver wire players) in the fantasy league.

            Returns up to 50 free agents with name, position, NBA team,
            percent_owned, and ownership_type. Results are cached in Postgres
            and automatically refreshed when new transactions (roster moves)
            are detected.

            Returns:
                JSON string with list of free agent player objects
            """
            return _ww_fetch()

        @tool
        def refresh_waiver_wire_cache() -> str:
            """
            Force a fresh fetch of waiver wire players from Yahoo, bypassing the cache.

            Use this when you suspect the cached data is incorrect.

            Returns:
                JSON string with updated list of free agent player objects
            """
            league_key = league_key_fn()
            if store is not None:
                try:
                    store.delete(league_key)
                except Exception as e:
                    logger.warning(f"Waiver wire cache delete failed: {e}")
            players = fetch_free_agents(limit=50)
            latest_tx_id = get_latest_tx_id()
            result = json.dumps(players, indent=2)
            if store is not None and latest_tx_id is not None:
                try:
                    store.put(league_key, players, latest_tx_id)
                except Exception as e:
                    logger.warning(f"Waiver wire store write failed: {e}")
            return result

        return [get_waiver_wire_players, refresh_waiver_wire_cache]
