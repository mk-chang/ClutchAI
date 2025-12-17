"""
Rotowire RSS feed tools for ClutchAI Agent.
"""

import json
from typing import Optional
from langchain_core.tools import tool

from .base import RSSTool


class RotowireRSSFeedTool(RSSTool):
    """
    Class for creating Rotowire RSS feed tools for LangChain agents.
    Provides tools for fetching NBA player news from Rotowire RSS feed.
    """
    
    def __init__(self, rss_url: Optional[str] = None, debug: bool = False):
        """
        Initialize RotowireRSSFeedTool.
        
        Args:
            rss_url: URL of the Rotowire RSS feed (default: None, should be provided from config)
            debug: Enable debug logging (default: False)
        """
        super().__init__(debug=debug)
        self.rss_url = rss_url
        if not self.rss_url:
            raise ValueError("rss_url is required. Provide it from config/tools_config.yaml or pass as parameter.")
    
    def create_get_nba_news_tool(self):
        """Create a tool for fetching NBA news from Rotowire RSS feed."""
        @tool(
            "get_rotowire_nba_news",
            description=(
                "Get the latest NBA player news from Rotowire RSS feed. "
                "Returns recent player updates including injuries, performance updates, and roster changes. "
                "Use this to get up-to-date fantasy basketball news for all players."
            )
        )
        def get_rotowire_nba_news(limit: Optional[int] = None) -> str:
            """
            Get NBA news from Rotowire RSS feed.
            
            Args:
                limit: Maximum number of news items to return (default: all items)
            
            Returns:
                Formatted JSON string with news items including title, description, link, and published date
            """
            try:
                feed_data = self._parse_feed(self.rss_url)
                
                entries = feed_data['entries']
                if limit and limit > 0:
                    entries = entries[:limit]
                
                result = {
                    'feed_title': feed_data['title'],
                    'feed_link': feed_data['link'],
                    'total_items': len(entries),
                    'items': entries
                }
                
                return f"Rotowire NBA News ({len(entries)} items):\n{self._format_response(result)}"
            except Exception as e:
                return f"Failed to fetch Rotowire NBA news: {e}"
        
        return get_rotowire_nba_news
    
    def get_all_tools(self):
        """
        Get all available Rotowire RSS tools.
        
        Returns:
            List of all LangChain tool instances
        """
        try:
            return [
                self.create_get_nba_news_tool(),
            ]
        except Exception as e:
            from logger import get_logger
            logger = get_logger(__name__)
            logger.warning(f"Rotowire RSS tools not available: {e}")
            return []

