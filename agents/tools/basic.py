"""
Basic utility tools for ClutchAI Agent.
Provides fundamental tools like date and time retrieval.
"""

from datetime import datetime
from typing import List
from langchain_core.tools import tool, BaseTool as LangChainBaseTool

from .base import ClutchAITool


class BasicTool(ClutchAITool):
    """
    Class for creating basic utility tools for LangChain agents.
    Provides fundamental tools like date and time retrieval.
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize BasicTool.
        
        Args:
            debug: Enable debug logging (default: False)
        """
        super().__init__(debug=debug)
    
    def get_all_tools(self) -> List[LangChainBaseTool]:
        """
        Get all available basic utility tools.
        
        Returns:
            List of all LangChain tool instances
        """
        return [
            self.create_get_current_datetime_tool(),
        ]
    
    def create_get_current_datetime_tool(self):
        """Create a tool for retrieving the current date and time."""
        @tool(
            "get_current_datetime",
            description="Get the current date and time in ISO format (YYYY-MM-DD HH:MM:SS). Use this when you need to know the current date and/or time. Always use this tool before calling date-based tools like get_team_roster_player_info_by_date() or get_player_stats_by_date() to get the current date in YYYY-MM-DD format."
        )
        def get_current_datetime() -> str:
            """Get the current date and time in ISO format."""
            try:
                current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                return f"Current date and time: {current_datetime}"
            except Exception as e:
                return f"Error getting current date and time: {e}"
        
        return get_current_datetime

