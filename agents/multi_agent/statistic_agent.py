"""
Statistic Agent for Multi-Agent System

This agent specializes in gathering NBA statistics and game data:
- Player statistics and performance metrics
- Team statistics and game logs
- Game data and scoreboards
- Career statistics and splits
"""

from pathlib import Path
from typing import Optional, List

from agents.multi_agent.base_agent import BaseAgent
from agents.tools.nba_api import nbaAPITool

logger = None  # Will be set by BaseAgent


class StatisticAgent(BaseAgent):
    """
    Statistic Agent that specializes in NBA API statistics and game data.
    """
    
    def _get_config_section(self) -> str:
        """Get the configuration section name for this agent."""
        return 'statistic'
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt if not found in config."""
        return """You are a statistics specialist agent for fantasy basketball analysis.
Your role is to gather comprehensive NBA statistics and game data:
- Player statistics: current season, career, game logs, splits
- Team statistics: team stats, game logs, performance metrics
- Game data: scoreboards, live scores, game information
- Player and team information: rosters, details, metadata

When given a research task, use all relevant NBA API tools to gather statistical data.
Be thorough and provide structured data that can be easily analyzed."""
    
    def _create_tools(self) -> List:
        """Create NBA API tools."""
        # Start with base tools (includes BasicTool)
        tools = list(super()._create_base_tools())
        
        # NBA API tools
        try:
            nba_tool = nbaAPITool(debug=self.debug)
            tools.extend(nba_tool.get_all_tools())
            self.logger.debug("NBA API tools loaded")
        except Exception as e:
            self.logger.warning(f"NBA API tools not available: {e}")
        
        self.logger.info(f"Statistic Agent initialized with {len(tools)} tools")
        return tools

