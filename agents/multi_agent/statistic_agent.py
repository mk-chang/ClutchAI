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

Z-scores: pV, rV, aV, sV, bV, pts3V (=three_v), toV, fgV, ftV.
value = sum of all 9 z-scores. pv = Yahoo points league value vs replacement.

For in-season analysis, prefer DB tools over live NBA API tools:
- get_recent_form(player_id, n_games=10) — hot/cold streaks, waiver wire form
- get_schedule_density(team_abbreviation, days=7) — games remaining, matchup density
- get_season_trends(player_id) — monthly trajectory, buy-low/sell-high
- query_stats_db(sql) — custom cross-table queries

opponent_defense_rankings ranks: 1=best defense (hardest matchup), 30=worst defense (easiest matchup).

Use live NBA API tools only for data not in the DB: live box scores, play-by-play, career stats."""

    def _create_tools(self) -> List:
        tools = list(super()._create_base_tools())

        try:
            tools.extend(PlayerStatsTool(debug=self.debug).get_all_tools())
            self.logger.debug("PlayerStatsTool loaded")
        except Exception as e:
            self.logger.warning(f"PlayerStatsTool not available: {e}")

        try:
            from agents.tools.player_stats_db import PlayerStatsDbTool
            tools.extend(PlayerStatsDbTool(debug=self.debug).get_all_tools())
            self.logger.debug("PlayerStatsDbTool loaded")
        except Exception as e:
            self.logger.warning(f"PlayerStatsDbTool not available: {e}")

        try:
            tools.extend(nbaAPITool(debug=self.debug).get_all_tools())
            self.logger.debug("NBA API tools loaded")
        except Exception as e:
            self.logger.warning(f"NBA API tools not available: {e}")

        self.logger.info(f"Statistic Agent initialized with {len(tools)} tools")
        return tools
