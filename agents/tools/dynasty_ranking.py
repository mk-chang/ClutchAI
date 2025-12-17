"""
Hashtag Basketball Dynasty Rankings tools for ClutchAI Agent.
"""

import json
import re
import pandas as pd
import requests
from io import StringIO
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from langchain_core.tools import tool, BaseTool as LangChainBaseTool

from .base import ClutchAITool
from logger import get_logger

logger = get_logger(__name__)


class DynastyRankingTool(ClutchAITool):
    """
    Class for creating Hashtag Basketball Dynasty Rankings tools for LangChain agents.
    Provides tools for scraping and querying dynasty rankings data.
    """
    
    def __init__(self, url: str = "https://hashtagbasketball.com/fantasy-basketball-dynasty-rankings", 
                 cache_duration_hours: int = 24, debug: bool = False):
        """
        Initialize DynastyRankingTool.
        
        Args:
            url: URL of the dynasty rankings page
            cache_duration_hours: How long to cache the rankings data in hours (default: 24)
            debug: Enable debug logging (default: False)
        """
        super().__init__(debug=debug)
        self.url = url
        self.cache_duration_hours = cache_duration_hours
        self._rankings_df: Optional[pd.DataFrame] = None
        self._last_fetch: Optional[datetime] = None
    
    def _scrape_rankings_table(self) -> pd.DataFrame:
        """
        Scrape the dynasty rankings table from Hashtag Basketball.
        
        Returns:
            DataFrame with rankings data
            
        Raises:
            Exception: If scraping fails
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            response = requests.get(self.url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # Parse all tables using pandas
            tables = pd.read_html(StringIO(response.text))
            
            # Find the main rankings table (should have many rows)
            for df in tables:
                if len(df) > 50:  # Main rankings table should have 700+ rows
                    logger.info(f"Successfully scraped dynasty rankings: {len(df)} players")
                    return df
            
            raise ValueError("Could not find main rankings table")
            
        except Exception as e:
            logger.error(f"Error scraping dynasty rankings: {e}")
            raise
    
    def _get_rankings_df(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get the rankings DataFrame, using cache if available and fresh.
        
        Args:
            force_refresh: Force a fresh scrape even if cache is valid
            
        Returns:
            DataFrame with rankings data
        """
        # Check if we need to refresh the cache
        should_refresh = (
            force_refresh or
            self._rankings_df is None or
            self._last_fetch is None or
            (datetime.now() - self._last_fetch) > timedelta(hours=self.cache_duration_hours)
        )
        
        if should_refresh:
            logger.info(f"Fetching fresh dynasty rankings from {self.url}")
            self._rankings_df = self._scrape_rankings_table()
            self._last_fetch = datetime.now()
        
        return self._rankings_df
    
    def _normalize_player_name(self, name: str) -> str:
        """
        Normalize player name for matching (remove extra spaces, lowercase, etc.).
        
        Args:
            name: Player name to normalize
            
        Returns:
            Normalized player name
        """
        # Remove leading/trailing whitespace and convert to lowercase
        normalized = name.strip().lower()
        # Replace multiple spaces with single space
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized
    
    def _find_player(self, player_name: str) -> Optional[pd.Series]:
        """
        Find a player in the rankings by name.
        
        Args:
            player_name: Name of the player to search for
            
        Returns:
            Series with player data if found, None otherwise
        """
        df = self._get_rankings_df()
        
        # Normalize the search name
        search_name = self._normalize_player_name(player_name)
        
        # Try exact match first on PLAYER column
        normalized_names = df['PLAYER'].apply(self._normalize_player_name)
        exact_match = normalized_names == search_name
        
        if exact_match.any():
            return df[exact_match].iloc[0]
        
        # Try partial match (contains)
        partial_match = normalized_names.str.contains(search_name, regex=False, na=False)
        
        if partial_match.any():
            # If multiple matches, return the first one (highest ranked)
            return df[partial_match].iloc[0]
        
        # Try searching in PLAYER.1 column (which contains full player info)
        if 'PLAYER.1' in df.columns:
            player1_normalized = df['PLAYER.1'].apply(self._normalize_player_name)
            partial_match = player1_normalized.str.contains(search_name, regex=False, na=False)
            
            if partial_match.any():
                return df[partial_match].iloc[0]
        
        return None
    
    def _format_player_data(self, player_row: pd.Series) -> Dict[str, Any]:
        """
        Format a player row into a structured dictionary.
        
        Args:
            player_row: Series with player data
            
        Returns:
            Dictionary with formatted player data
        """
        player_data = {
            'rank': player_row.get('RANK', 'N/A'),
            'player_name': player_row.get('PLAYER', 'N/A'),
            'age': player_row.get('AGE', 'N/A'),
            'team': player_row.get('TEAM', 'N/A'),
            'position': player_row.get('POS', 'N/A'),
            'comments': player_row.get('COMMENTS', 'N/A'),
        }
        
        # Extract additional info from PLAYER.1 column if available
        if 'PLAYER.1' in player_row and pd.notna(player_row['PLAYER.1']):
            player_data['detailed_info'] = player_row['PLAYER.1']
        
        # Try to parse stats from COMMENTS if available
        if 'COMMENTS' in player_row and pd.notna(player_row['COMMENTS']):
            comments = str(player_row['COMMENTS'])
            player_data['comments'] = comments
        
        return player_data
    
    def create_get_player_dynasty_rank_tool(self):
        """Create a tool for getting a player's dynasty ranking."""
        @tool(
            "get_player_dynasty_rank",
            description=(
                "Get dynasty ranking information for a specific NBA player from Hashtag Basketball. "
                "Returns the player's rank, age, team, position, and comments. "
                "You can search by full name or partial name (e.g., 'LeBron', 'Stephen Curry', 'Luka')."
            )
        )
        def get_player_dynasty_rank(player_name: str) -> str:
            """
            Get dynasty ranking for a player.
            
            Args:
                player_name: Name of the player to search for
            
            Returns:
                Formatted JSON string with player ranking data
            """
            try:
                player_row = self._find_player(player_name)
                
                if player_row is None:
                    return f"Player '{player_name}' not found in dynasty rankings. Try searching with a different name or check the spelling."
                
                player_data = self._format_player_data(player_row)
                return f"Player dynasty ranking data:\n{self._format_response(player_data)}"
                
            except Exception as e:
                logger.error(f"Error getting player dynasty rank: {e}")
                return f"Failed to get dynasty ranking for '{player_name}': {str(e)}"
        
        return get_player_dynasty_rank
    
    def create_get_all_rankings_tool(self):
        """Create a tool for getting all dynasty rankings."""
        @tool(
            "get_all_dynasty_rankings",
            description=(
                "Get all dynasty rankings from Hashtag Basketball. "
                "Returns the complete list of all players with their rankings, age, team, position, and comments."
            )
        )
        def get_all_dynasty_rankings() -> str:
            """
            Get all dynasty rankings.
            
            Returns:
                Formatted JSON string with all players' ranking data
            """
            try:
                df = self._get_rankings_df()
                
                # Format all results
                results = []
                for _, row in df.iterrows():
                    results.append(self._format_player_data(row))
                
                return f"All dynasty rankings ({len(results)} players):\n{self._format_response(results)}"
                
            except Exception as e:
                logger.error(f"Error getting all rankings: {e}")
                return f"Failed to get all dynasty rankings: {str(e)}"
        
        return get_all_dynasty_rankings
    
    def create_get_top_rankings_tool(self):
        """Create a tool for getting top X dynasty rankings."""
        @tool(
            "get_top_dynasty_rankings",
            description=(
                "Get the top X players from dynasty rankings. "
                "Returns the highest ranked players with their rank, age, team, position, and comments."
            )
        )
        def get_top_dynasty_rankings(top_n: int = 10) -> str:
            """
            Get top N dynasty rankings.
            
            Args:
                top_n: Number of top players to return (default: 10)
            
            Returns:
                Formatted JSON string with top players' ranking data
            """
            try:
                df = self._get_rankings_df()
                
                # Get top N players (already sorted by rank)
                top_players = df.head(top_n)
                
                # Format results
                results = []
                for _, row in top_players.iterrows():
                    results.append(self._format_player_data(row))
                
                return f"Top {len(results)} dynasty rankings:\n{self._format_response(results)}"
                
            except Exception as e:
                logger.error(f"Error getting top rankings: {e}")
                return f"Failed to get top dynasty rankings: {str(e)}"
        
        return get_top_dynasty_rankings
    
    def create_get_rankings_by_position_tool(self):
        """Create a tool for getting dynasty rankings filtered by position."""
        @tool(
            "get_dynasty_rankings_by_position",
            description=(
                "Get dynasty rankings filtered by position. "
                "Returns all players at the specified position with their rank, age, team, and comments. "
                "Common positions: PG, SG, SF, PF, C (or combinations like PG/SG, SF/PF, etc.)"
            )
        )
        def get_dynasty_rankings_by_position(position: str) -> str:
            """
            Get dynasty rankings by position.
            
            Args:
                position: Position to filter by (e.g., 'PG', 'SG', 'SF', 'PF', 'C', 'PG/SG', etc.)
            
            Returns:
                Formatted JSON string with players' ranking data for the specified position
            """
            try:
                df = self._get_rankings_df()
                
                # Normalize position for matching (case-insensitive)
                position_normalized = position.strip().upper()
                
                # Filter by position - check if position string is contained in POS column
                if 'POS' in df.columns:
                    # Handle both exact matches and position combinations (e.g., "PG/SG" contains "PG")
                    position_filter = df['POS'].astype(str).str.contains(position_normalized, regex=False, na=False)
                    filtered_df = df[position_filter]
                else:
                    return "Position column (POS) not found in rankings data."
                
                if len(filtered_df) == 0:
                    return f"No players found at position '{position}' in dynasty rankings."
                
                # Format results
                results = []
                for _, row in filtered_df.iterrows():
                    results.append(self._format_player_data(row))
                
                return f"Found {len(results)} player(s) at position '{position}':\n{self._format_response(results)}"
                
            except Exception as e:
                logger.error(f"Error getting rankings by position: {e}")
                return f"Failed to get dynasty rankings by position: {str(e)}"
        
        return get_dynasty_rankings_by_position
    
    def create_refresh_rankings_tool(self):
        """Create a tool for refreshing the rankings cache."""
        @tool(
            "refresh_dynasty_rankings",
            description=(
                "Refresh the dynasty rankings data from Hashtag Basketball. "
                "Use this if you want to get the most up-to-date rankings. "
                "By default, rankings are cached for 24 hours."
            )
        )
        def refresh_dynasty_rankings() -> str:
            """Refresh the dynasty rankings cache."""
            try:
                self._rankings_df = self._scrape_rankings_table()
                self._last_fetch = datetime.now()
                num_players = len(self._rankings_df)
                return f"Successfully refreshed dynasty rankings. Found {num_players} players. Last updated: {self._last_fetch.strftime('%Y-%m-%d %H:%M:%S')}"
            except Exception as e:
                logger.error(f"Error refreshing rankings: {e}")
                return f"Failed to refresh dynasty rankings: {str(e)}"
        
        return refresh_dynasty_rankings
    
    def get_all_tools(self) -> List[LangChainBaseTool]:
        """
        Get all available tools for dynasty rankings.
        
        Returns:
            List of all LangChain tool instances (empty list if initialization failed)
        """
        try:
            tools = [
                self.create_get_player_dynasty_rank_tool(),
                self.create_get_all_rankings_tool(),
                self.create_get_top_rankings_tool(),
                self.create_get_rankings_by_position_tool(),
                self.create_refresh_rankings_tool(),
            ]
            return tools
        except Exception as e:
            logger.warning(f"Dynasty Ranking tools not available: {e}")
            return []
