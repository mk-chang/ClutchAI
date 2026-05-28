"""
LangChain tool: basketball_monster_stats
Queries per-game, totals, or per-36 stats + z-scores + value metrics.
"""

import json
from typing import Optional

from langchain_core.tools import tool
from sqlalchemy import text

from agents.tools.base import ClutchAITool
from data.postgres.connection import PostgresConnection

_TABLES = {
    'pg':    'bball_monsters_player_stats_pg',
    'total': 'bball_monsters_player_stats_total',
    'p36':   'bball_monsters_player_stats_p36',
}


class PlayerStatsTool(ClutchAITool):
    """Query NBA player stats (pg/total/p36) with z-scores and value metrics."""

    def __init__(
        self,
        connection: Optional[PostgresConnection] = None,
        season: str = '2025-26',
        debug: bool = False,
    ):
        super().__init__(debug=debug)
        self.connection = connection or PostgresConnection()
        self.season = season

    def _query(
        self,
        season: str,
        stat_type: str = 'pg',
        player_name: Optional[str] = None,
        team: Optional[str] = None,
        limit: int = 20,
    ) -> list:
        table = _TABLES[stat_type]
        sql = text(f"""
            SELECT *
            FROM {table}
            WHERE season = :season
              AND (:player_name IS NULL OR player_name ILIKE :player_name)
              AND (:team IS NULL OR team_abbreviation = :team)
            ORDER BY rv DESC NULLS LAST
            LIMIT :limit
        """)
        engine = self.connection.get_engine()
        with engine.connect() as conn:
            result = conn.execute(sql, {
                'season':      season,
                'player_name': f'%{player_name}%' if player_name else None,
                'team':        team.upper() if team else None,
                'limit':       limit,
            })
            rows = result.fetchall()
            keys = list(result.keys())
        return [dict(zip(keys, row)) for row in rows]

    def create_basketball_monster_stats_tool(self):
        query_fn = self._query
        default_season = self.season

        @tool(
            "basketball_monster_stats",
            description=(
                "Query NBA player stats with z-scores and value metrics (Basketball Monster style). "
                "stat_type: 'pg' (per-game, default), 'total' (season totals), 'p36' (per-36 minutes). "
                "Returns stats, z_pts/z_reb/z_ast/z_stl/z_blk/z_3ptm/z_tov/z_fg/z_ft, "
                "rv (roto value), three_v (3-point value), pv (Yahoo points value). "
                "Filter by player_name (partial) or team abbreviation (e.g. 'LAL'). "
                "season_override: '2023-24', '2024-25', or '2025-26'."
            ),
        )
        def basketball_monster_stats(
            player_name: Optional[str] = None,
            team: Optional[str] = None,
            stat_type: str = 'pg',
            season_override: Optional[str] = None,
            limit: int = 20,
        ) -> str:
            season = season_override or default_season
            if stat_type not in _TABLES:
                return f"Invalid stat_type '{stat_type}'. Use: pg, total, p36."
            results = query_fn(
                season=season, stat_type=stat_type,
                player_name=player_name, team=team, limit=limit,
            )
            if not results:
                msg = f"No players found for season {season} [{stat_type}]"
                if player_name:
                    msg += f" matching '{player_name}'"
                if team:
                    msg += f" on team '{team}'"
                return msg
            return json.dumps(results, default=str)

        return basketball_monster_stats

    def get_all_tools(self):
        return [self.create_basketball_monster_stats_tool()]
