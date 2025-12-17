"""
Yahoo Fantasy News tools for ClutchAI Agent using Firecrawl.
"""

import os
import json
from typing import Optional, List
from langchain_core.tools import tool

from .base import FirecrawlTool


class FantasyNewsTool(FirecrawlTool):
    """
    Class for creating Yahoo Fantasy News tools for LangChain agents.
    Provides tools for scraping Yahoo Fantasy NBA news using Firecrawl.
    """
    
    def __init__(self, api_key: Optional[str] = None, urls: Optional[List[str]] = None, map_limit: int = 200, debug: bool = False):
        """
        Initialize FantasyNewsTool.
        
        Args:
            api_key: Firecrawl API key (or from env FIRECRAWL_API_KEY)
            urls: List of URLs to map for discovering Yahoo Fantasy NBA news links (from config/tools_config.yaml)
            map_limit: Default limit for map operations (default: 200)
            debug: Enable debug logging (default: False)
        """
        super().__init__(api_key=api_key, debug=debug)
        self.urls = urls or []
        self.map_limit = map_limit
    
    def create_scrape_url_tool(self):
        """Create a generic tool for scraping any URL."""
        @tool(
            "scrape_url",
            description=(
                "Scrape content from any URL using Firecrawl. "
                "Returns the scraped content in markdown format with metadata."
            )
        )
        def scrape_url(url: str) -> str:
            """
            Scrape content from a URL.
            
            Args:
                url: URL to scrape
            
            Returns:
                Formatted JSON string with scraped content in markdown format including metadata.
                Includes date_published field if available in the response.
            """
            try:
                # Scrape the URL using Firecrawl v2 API
                # Documentation: https://docs.firecrawl.dev/sdks/python#scraping-a-url
                # Always use markdown format
                result = self.app.scrape(url, formats=['markdown'])
                
                # Handle response - could be dict or object with attributes
                if hasattr(result, 'markdown') or (hasattr(result, 'get') and callable(getattr(result, 'get'))):
                    # Response is an object with attributes (or dict-like with get method)
                    try:
                        if hasattr(result, 'markdown'):
                            markdown = result.markdown
                            title = getattr(result, 'title', '')
                            description = getattr(result, 'description', '')
                            links = getattr(result, 'links', [])
                            # Try to get date/time from various possible fields
                            date_published = (
                                getattr(result, 'publishedDate', None) or
                                getattr(result, 'datePublished', None) or
                                getattr(result, 'date', None) or
                                getattr(result, 'timestamp', None) or
                                (hasattr(result, 'metadata') and getattr(result.metadata, 'publishedDate', None)) or
                                (hasattr(result, 'metadata') and getattr(result.metadata, 'datePublished', None))
                            )
                        else:
                            # Try as dictionary
                            markdown = result.get('markdown', '')
                            title = result.get('title', '')
                            description = result.get('description', '')
                            links = result.get('links', [])
                            # Try to get date/time from various possible fields
                            metadata = result.get('metadata', {})
                            date_published = (
                                result.get('publishedDate') or
                                result.get('datePublished') or
                                result.get('date') or
                                result.get('timestamp') or
                                (metadata.get('publishedDate') if isinstance(metadata, dict) else None) or
                                (metadata.get('datePublished') if isinstance(metadata, dict) else None)
                            )
                        
                        formatted_result = {
                            'url': url,
                            'content': markdown,
                            'title': title,
                            'description': description,
                            'links': links,
                            'date': date_published,  # Date from the scraped page, if available
                        }
                        
                        # Also add date_published for backward compatibility
                        if date_published:
                            formatted_result['date_published'] = date_published
                        
                        return f'Scraped content from {url}:\n{self._format_response(formatted_result)}'
                    except Exception as e:
                        # If we can't parse it, return raw result
                        return f'Scraped content from {url}:\n{self._format_response(result)}'
                else:
                    # Unknown format, return as-is
                    return f'Scraped content from {url}:\n{self._format_response(result)}'
            except Exception as e:
                return f"Failed to scrape URL: {e}"
        return scrape_url
    
    def create_map_all_configured_urls_tool(self):
        """Create a tool for mapping all configured Yahoo Fantasy NBA news URLs to discover links."""
        @tool(
            "map_all_yahoo_fantasy_nba_news",
            description=(
                "Map all configured Yahoo Fantasy NBA news URLs from config/tools_config.yaml to discover available links. "
                "Returns discovered URLs with titles and descriptions. Use this first to understand what content is available, "
                "then use scrape_url on specific URLs as needed. Much faster than scraping since it only discovers URLs."
            )
        )
        def map_all_yahoo_fantasy_nba_news(limit: Optional[int] = None, search: Optional[str] = None) -> str:
            """
            Map all configured Yahoo Fantasy NBA news URLs to discover links.
            
            Args:
                limit: Maximum number of links to return per URL (default: uses map_limit from tool initialization)
                search: Optional search term to filter URLs by keyword
            
            Returns:
                Formatted JSON string with discovered URLs, titles, and descriptions from all configured URLs
            """
            if limit is None:
                limit = self.map_limit
            if not self.urls:
                return "No URLs configured. Add URLs to tools_config.yaml under 'yahoo_fantasy_news_urls'."
            
            results = []
            for url in self.urls:
                try:
                    # Map the URL using Firecrawl v2 API to discover links
                    if search:
                        result = self.app.map(url, limit=limit, search=search)
                    else:
                        result = self.app.map(url, limit=limit)
                    
                    if result and 'links' in result:
                        links = result['links']
                        # Preserve all metadata from Firecrawl response
                        formatted_result = {
                            'url': url,
                            'search': search,
                            'total_links': len(links),
                            'links': links,
                        }
                        # Include any other metadata from the Firecrawl response
                        for key, value in result.items():
                            if key not in ['links']:  # Already included above
                                formatted_result[f'firecrawl_{key}'] = value
                        results.append(formatted_result)
                    else:
                        results.append({
                            'url': url,
                            'error': 'No links found',
                            'raw_result': result
                        })
                except Exception as e:
                    results.append({
                        'url': url,
                        'error': str(e)
                    })
            
            return f'Mapped URLs from {len(self.urls)} configured Yahoo Fantasy NBA news sources:\n{self._format_response(results)}'
        
        return map_all_yahoo_fantasy_nba_news
    
    def create_map_url_tool(self):
        """Create a tool for mapping a website to discover URLs."""
        @tool(
            "map_url",
            description=(
                "Map a website to quickly discover and list URLs without extracting content. "
                "Returns URLs with titles and descriptions. Use this first to understand site structure "
                "or select specific pages to scrape later."
            )
        )
        def map_url(
            url: str,
            limit: Optional[int] = None,
            search: Optional[str] = None
        ) -> str:
            """
            Map a website to discover URLs.
            
            Args:
                url: Base URL to map
                limit: Maximum number of links to return (default: uses map_limit from tool initialization)
                search: Optional search term to filter URLs by keyword
            
            Returns:
                Formatted JSON string with discovered URLs, titles, and descriptions
            """
            if limit is None:
                limit = self.map_limit
            try:
                # Map the URL using Firecrawl v2 API
                if search:
                    result = self.app.map(url, limit=limit, search=search)
                else:
                    result = self.app.map(url, limit=limit)
                
                if result and 'links' in result:
                    links = result['links']
                    # Preserve all metadata from Firecrawl response
                    formatted_result = {
                        'url': url,
                        'search': search,
                        'total_links': len(links),
                        'links': links,
                    }
                    # Include any other metadata from the Firecrawl response
                    for key, value in result.items():
                        if key not in ['links']:  # Already included above
                            formatted_result[f'firecrawl_{key}'] = value
                    return f'Mapped URLs from {url}:\n{self._format_response(formatted_result)}'
                else:
                    return f'Mapped URLs from {url}:\n{self._format_response(result)}'
            except Exception as e:
                return f"Failed to map URL: {e}"
        
        return map_url
    
    def get_all_tools(self):
        """
        Get all available Yahoo Fantasy News tools.
        
        Returns:
            List of all LangChain tool instances (empty list if initialization failed)
        """
        try:
            tools = [
                self.create_scrape_url_tool(),
                self.create_map_url_tool(),
            ]
            
            # Only add map_all tool if URLs are configured
            if self.urls:
                tools.append(self.create_map_all_configured_urls_tool())
            
            return tools
        except (ValueError, ImportError) as e:
            from logger import get_logger
            logger = get_logger(__name__)
            logger.warning(f"Yahoo Fantasy News tools not available: {e}")
            return []
