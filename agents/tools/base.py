"""
Base class for ClutchAI Agent tools.
"""

import os
import json
from abc import ABC, abstractmethod
from typing import List, Optional
from langchain_core.tools import BaseTool as LangChainBaseTool

from logger import get_logger

logger = get_logger(__name__)

# Import Firecrawl
try:
    from firecrawl import Firecrawl
    try:
        import firecrawl
        FIRECRAWL_VERSION = getattr(firecrawl, '__version__', None)
    except (ImportError, AttributeError):
        FIRECRAWL_VERSION = None
except ImportError:
    Firecrawl = None
    FIRECRAWL_VERSION = None


class ClutchAITool(ABC):
    """
    Base class for creating tool classes for LangChain agents.
    
    Subclasses should:
    1. Implement __init__ with their specific initialization
    2. Override _format_response if needed for custom formatting
    3. Implement get_all_tools to return a list of tool instances
    4. Create individual tool methods using @tool decorator
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize ClutchAITool.
        
        Args:
            debug: Enable debug logging (default: False)
        """
        self.debug = debug
        self.logger = get_logger(self.__class__.__name__)
    
    def _format_response(self, data) -> str:
        """
        Helper method to format API response data as JSON string.
        
        This is a base implementation that handles common cases.
        Subclasses can override this for API-specific formatting needs.
        
        Args:
            data: Response data from API call
            
        Returns:
            Formatted JSON string representation of the data
        """
        try:
            # Handle pandas DataFrames
            if hasattr(data, 'to_dict'):
                return json.dumps(data.to_dict('records'), default=str, indent=2)
            # Handle objects with get_dict() method (e.g., nba_api responses)
            elif hasattr(data, 'get_dict'):
                return json.dumps(data.get_dict(), default=str, indent=2)
            # Handle objects with get_json() method
            elif hasattr(data, 'get_json'):
                return data.get_json()
            # Handle objects with __dict__
            elif hasattr(data, '__dict__'):
                return json.dumps(data.__dict__, default=str, indent=2)
            # Handle dict/list
            elif isinstance(data, (dict, list)):
                return json.dumps(data, default=str, indent=2)
            else:
                return str(data)
        except Exception as e:
            return f"Error formatting response: {str(e)}\nRaw data: {str(data)}"
    
    @abstractmethod
    def get_all_tools(self) -> List[LangChainBaseTool]:
        """
        Get all available tools for this tool class.
        
        Returns:
            List of all LangChain tool instances
        """
        pass


class FirecrawlTool(ClutchAITool):
    """
    Base class for Firecrawl-based tools that use the Firecrawl API.
    
    This class provides common functionality for tools that interact with Firecrawl:
    - Firecrawl initialization and API key management
    - Common response formatting for Firecrawl responses
    - Shared utility methods
    
    Subclasses should:
    1. Call super().__init__(api_key, debug) in their __init__
    2. Implement get_all_tools to return their specific tool instances
    3. Use self.app to access the Firecrawl instance
    """
    
    def __init__(self, api_key: Optional[str] = None, debug: bool = False):
        """
        Initialize FirecrawlTool.
        
        Args:
            api_key: Firecrawl API key (or from env FIRECRAWL_API_KEY)
            debug: Enable debug logging (default: False)
        """
        super().__init__(debug=debug)
        
        if Firecrawl is None:
            raise ImportError(
                "firecrawl-py is not installed. Please install it with: pip install firecrawl-py"
            )
        
        self.api_key = api_key or os.environ.get('FIRECRAWL_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Firecrawl API key is required. Set FIRECRAWL_API_KEY env var or pass api_key parameter."
            )
        
        self.app = Firecrawl(api_key=self.api_key)
        
        # Log version information in debug mode
        if debug:
            version_str = f" (version {FIRECRAWL_VERSION})" if FIRECRAWL_VERSION else ""
            logger.debug(f"Firecrawl initialized{version_str}")
            if FIRECRAWL_VERSION is None:
                logger.debug("Note: Could not detect Firecrawl package version")
    
    def _format_response(self, data) -> str:
        """
        Helper method to format Firecrawl response data as JSON string.
        
        This implementation handles Firecrawl-specific response formats.
        Subclasses can override if they need custom formatting.
        
        Args:
            data: Response data from Firecrawl API call
            
        Returns:
            Formatted JSON string representation of the data
        """
        try:
            # Handle dict/list
            if isinstance(data, (dict, list)):
                return json.dumps(data, default=str, indent=2)
            # Handle objects with __dict__
            elif hasattr(data, '__dict__'):
                return json.dumps(data.__dict__, default=str, indent=2)
            else:
                return str(data)
        except Exception as e:
            return f"Error formatting response: {str(e)}\nRaw data: {str(data)}"
    
    @abstractmethod
    def get_all_tools(self) -> List[LangChainBaseTool]:
        """
        Get all available tools for this Firecrawl-based tool class.
        
        Returns:
            List of all LangChain tool instances
        """
        pass


class RSSTool(ClutchAITool):
    """
    Base class for RSS feed-based tools.
    
    This class provides common functionality for tools that parse RSS feeds:
    - RSS feed parsing using feedparser
    - Common response formatting for RSS items
    - Shared utility methods
    
    Subclasses should:
    1. Call super().__init__(debug) in their __init__
    2. Implement get_all_tools to return their specific tool instances
    3. Use self._parse_feed(url) to parse RSS feeds
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize RSSTool.
        
        Args:
            debug: Enable debug logging (default: False)
        """
        super().__init__(debug=debug)
        
        try:
            import feedparser
            self.feedparser = feedparser
        except ImportError:
            raise ImportError(
                "feedparser is not installed. Please install it with: pip install feedparser"
            )
    
    def _parse_feed(self, url: str) -> dict:
        """
        Parse an RSS feed from a URL.
        
        Args:
            url: URL of the RSS feed
            
        Returns:
            Dictionary with parsed feed data including entries
        """
        try:
            feed = self.feedparser.parse(url)
            
            if feed.bozo and feed.bozo_exception:
                self.logger.warning(f"RSS feed parsing warning: {feed.bozo_exception}")
            
            return {
                'title': feed.feed.get('title', ''),
                'link': feed.feed.get('link', ''),
                'description': feed.feed.get('description', ''),
                'entries': [
                    {
                        'title': entry.get('title', ''),
                        'link': entry.get('link', ''),
                        'description': entry.get('description', ''),
                        'published': entry.get('published', ''),
                        'published_parsed': entry.get('published_parsed'),
                        'guid': entry.get('id', entry.get('link', '')),
                    }
                    for entry in feed.entries
                ],
                'total_entries': len(feed.entries),
            }
        except Exception as e:
            self.logger.error(f"Error parsing RSS feed {url}: {e}")
            raise
    
    def _format_response(self, data) -> str:
        """
        Helper method to format RSS feed data as JSON string.
        
        Args:
            data: Response data from RSS feed parsing
            
        Returns:
            Formatted JSON string representation of the data
        """
        return super()._format_response(data)
    
    @abstractmethod
    def get_all_tools(self) -> List[LangChainBaseTool]:
        """
        Get all available tools for this RSS-based tool class.
        
        Returns:
            List of all LangChain tool instances
        """
        pass

