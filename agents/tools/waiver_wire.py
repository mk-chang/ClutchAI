import json
from typing import List
from langchain_core.tools import tool

from .base import ClutchAITool
from logger import get_logger

logger = get_logger(__name__)

_BATCH_SIZE = 25


class WaiverWireTool(ClutchAITool):
    """Fetches free agents from Yahoo Fantasy API."""

    def __init__(self, query, debug: bool = False):
        super().__init__(debug=debug)
        self.query = query

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

    def get_all_tools(self) -> list:
        fetch_free_agents = self._fetch_free_agents

        @tool
        def get_waiver_wire_players() -> str:
            """
            Get available free agents (waiver wire players) in the fantasy league.

            Returns up to 50 free agents with name, position, NBA team,
            percent_owned, and ownership_type. Always fetches live from Yahoo.

            Returns:
                JSON string with list of free agent player objects
            """
            return json.dumps(fetch_free_agents(limit=50), indent=2)

        return [get_waiver_wire_players]
