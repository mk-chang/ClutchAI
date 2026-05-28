"""
Statistic Agent — NBA stats and game data.

basketball_monster_stats: per-game/total/per-36 stats with z-scores (z_pts, z_reb,
z_ast, z_stl, z_blk, z_3ptm, z_tov, z_fg, z_ft) and value metrics (rv, pv, three_v).
NBA API tools: career stats, game logs, live scores, splits.
"""

from typing import List

from agents.multi_agent.base_agent import BaseAgent
from agents.tools.nba_api import nbaAPITool
from agents.tools.player_stats import PlayerStatsTool

logger = None


class StatisticAgent(BaseAgent):

    def _get_config_section(self) -> str:
        return 'statistic'

    def _get_default_system_prompt(self) -> str:
        return """You are a statistics specialist for fantasy basketball analysis.

Use basketball_monster_stats for player stats with value metrics:
- stat_type='pg'    → per-game averages + z-scores + rv, pv, three_v
- stat_type='total' → season totals + z-scores + rv, three_v
- stat_type='p36'   → per-36 minute rates + z-scores + rv, pv, three_v
Available seasons: 2023-24, 2024-25, 2025-26.

Z-scores: z_pts, z_reb, z_ast, z_stl, z_blk, z_3ptm (=three_v), z_tov, z_fg, z_ft.
rv = sum of all 9 z-scores. pv = Yahoo points league value vs replacement.

Use NBA API tools for career stats, game logs, live scores, and splits."""

    def _create_tools(self) -> List:
        tools = list(super()._create_base_tools())

        try:
            tools.extend(PlayerStatsTool(debug=self.debug).get_all_tools())
            self.logger.debug("PlayerStatsTool loaded")
        except Exception as e:
            self.logger.warning(f"PlayerStatsTool not available: {e}")

        try:
            tools.extend(nbaAPITool(debug=self.debug).get_all_tools())
            self.logger.debug("NBA API tools loaded")
        except Exception as e:
            self.logger.warning(f"NBA API tools not available: {e}")

        self.logger.info(f"Statistic Agent initialized with {len(tools)} tools")
        return tools
