"""
Yahoo Fantasy Agent for Multi-Agent System

This agent specializes in gathering data from Yahoo Fantasy API:
- League data, rosters, standings, matchups
- Player stats and ownership
- Team information
- Draft results and transactions
"""

from pathlib import Path
from typing import Optional, List
from yfpy.query import YahooFantasySportsQuery

from agents.multi_agent.base_agent import BaseAgent
from agents.tools.yahoo_api import YahooFantasyTool

logger = None  # Will be set by BaseAgent


class YahooFantasyAgent(BaseAgent):
    """
    Yahoo Fantasy Agent that specializes in Yahoo Fantasy API data.
    """
    
    def _get_config_section(self) -> str:
        """Get the configuration section name for this agent."""
        return 'yahoo_fantasy'
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt if not found in config."""
        return """You are a Yahoo Fantasy specialist agent for fantasy basketball analysis.
Your role is to gather comprehensive data from the Yahoo Fantasy API:
- League data: standings, settings, teams, players
- Team data: rosters, stats, matchups, draft results
- Player data: stats, ownership, draft analysis
- Transaction data: trades, waiver claims, adds/drops

When given a research task, use all relevant Yahoo Fantasy tools to gather data.
Be thorough and provide structured data that can be easily analyzed."""
    
    def _create_tools(self) -> List:
        """Create Yahoo Fantasy API tools."""
        # Start with base tools (includes BasicTool)
        tools = list(super()._create_base_tools())
        
        # Yahoo Fantasy API tools
        if self.query is None:
            self.logger.warning("YahooFantasySportsQuery not provided. Yahoo Fantasy tools will not be available.")
        else:
            try:
                yahoo_tool = YahooFantasyTool(
                    query=self.query,
                    debug=self.debug,
                )
                tools.extend(yahoo_tool.get_all_tools())
                self.logger.debug("Yahoo Fantasy tools loaded")
            except Exception as e:
                self.logger.warning(f"Yahoo Fantasy tools not available: {e}")
        
        self.logger.info(f"Yahoo Fantasy Agent initialized with {len(tools)} tools")
        return tools

