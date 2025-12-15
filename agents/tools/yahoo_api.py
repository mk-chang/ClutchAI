"""
Yahoo Fantasy Sports tools for ClutchAI Agent.
"""

import json
from typing import Union
from langchain_core.tools import tool
from yfpy.query import YahooFantasySportsQuery
from .base import ClutchAITool


class YahooFantasyTool(ClutchAITool):
    """
    Class for creating Yahoo Fantasy Sports tools for LangChain agents.
    Provides tools for all Yahoo Fantasy Sports API get_* methods.
    """
    
    def __init__(self, query: YahooFantasySportsQuery, debug: bool = False):
        """
        Initialize YahooFantasyTool with a YahooFantasySportsQuery instance.
        
        Args:
            query: YahooFantasySportsQuery instance
            debug: Enable debug logging (default: False)
        """
        super().__init__(debug=debug)
        self.query = query
 
    def _format_response(self, data) -> str:
        """Helper method to format API response data as JSON string."""
        try:
            if hasattr(data, '__dict__'):
                # Convert object to dict if possible
                # Handle KeyError when accessing attributes that don't exist
                try:
                    return json.dumps(data.__dict__, default=str, indent=2)
                except (KeyError, AttributeError, TypeError) as e:
                    # If __dict__ access fails, try to serialize safely
                    try:
                        # Try to get a safe dict representation
                        safe_dict = {}
                        for key in dir(data):
                            if not key.startswith('_'):
                                try:
                                    value = getattr(data, key)
                                    if not callable(value):
                                        safe_dict[key] = value
                                except (KeyError, AttributeError):
                                    pass
                        return json.dumps(safe_dict, default=str, indent=2)
                    except Exception:
                        return str(data)
            elif isinstance(data, (dict, list)):
                return json.dumps(data, default=str, indent=2)
            else:
                return str(data)
        except (KeyError, AttributeError, TypeError) as e:
            try:
                return str(data)
            except Exception:
                return f"Response data (unable to serialize: {type(e).__name__})"
        except Exception:
            return str(data)
    
    def _extract_team_id(self, team_key_or_id: str) -> int:
        """
        Extract team_id from team_key or return team_id if already numeric.
        
        Team key format: {game_id}.l.{league_id}.t.{team_id}
        Example: 466.l.58930.t.6 -> returns 6
        
        Args:
            team_key_or_id: Team key string or team_id (numeric string)
            
        Returns:
            Team ID as integer
        """
        if not team_key_or_id:
            raise ValueError("Team key or ID cannot be empty")
        
        # Remove any whitespace
        team_key_or_id = team_key_or_id.strip()
        
        # If it's just a number, return it as int
        if team_key_or_id.isdigit():
            return int(team_key_or_id)
        
        # Try to extract team_id from team_key format: game_id.l.league_id.t.team_id
        # Look for .t. pattern and extract the number after it
        if '.t.' in team_key_or_id:
            parts = team_key_or_id.split('.t.')
            if len(parts) == 2:
                team_id_part = parts[1].split('.')[0]  # Get the team_id part before any additional dots
                if team_id_part.isdigit():
                    return int(team_id_part)
        
        # Check for duplicate patterns (e.g., "466.l.58930.t.466.l.58930.t.6")
        # Split by common delimiters
        parts = team_key_or_id.replace('.', ' ').split()
        
        # Look for the pattern: game_id l league_id t team_id
        # If we see duplicates, extract the last valid sequence
        if len(parts) >= 5:
            # Try to find the last valid sequence
            # Pattern should be: number, 'l', number, 't', number
            for i in range(len(parts) - 4, -1, -1):
                if (i + 4 < len(parts) and 
                    parts[i].isdigit() and 
                    parts[i+1].lower() == 'l' and 
                    parts[i+2].isdigit() and 
                    parts[i+3].lower() == 't' and 
                    parts[i+4].isdigit()):
                    # Found valid pattern, return the team_id
                    return int(parts[i+4])
        
        raise ValueError(f"Could not extract team_id from: {team_key_or_id}")
    
    def _normalize_team_key(self, team_key: str) -> str:
        """
        Normalize and validate team key format.
        
        Team key format should be: {game_id}.l.{league_id}.t.{team_id}
        Example: 466.l.58930.t.6
        
        Args:
            team_key: Team key string (may be malformed)
            
        Returns:
            Normalized team key string
        """
        if not team_key:
            raise ValueError("Team key cannot be empty")
        
        # Remove any whitespace
        team_key = team_key.strip()
        
        # Check for duplicate patterns (e.g., "466.l.58930.t.466.l.58930.t.6")
        # Split by common delimiters
        parts = team_key.replace('.', ' ').split()
        
        # Look for the pattern: game_id l league_id t team_id
        # If we see duplicates, extract the last valid sequence
        if len(parts) >= 6:
            # Try to find the last valid sequence
            # Pattern should be: number, 'l', number, 't', number
            for i in range(len(parts) - 4, -1, -1):
                if (i + 4 < len(parts) and 
                    parts[i].isdigit() and 
                    parts[i+1].lower() == 'l' and 
                    parts[i+2].isdigit() and 
                    parts[i+3].lower() == 't' and 
                    parts[i+4].isdigit()):
                    # Found valid pattern
                    game_id = parts[i]
                    league_id = parts[i+2]
                    team_id = parts[i+4]
                    return f"{game_id}.l.{league_id}.t.{team_id}"
        
        # If no duplicates found, check if it's already in correct format
        if team_key.count('.l.') == 1 and team_key.count('.t.') == 1:
            return team_key
        
        # If it's just a number, try to construct from query's league key
        if team_key.isdigit():
            try:
                league_key = self.query.get_league_key()
                # Extract game_id and league_id from league_key (format: game_id.l.league_id)
                if '.l.' in league_key:
                    game_id, league_part = league_key.split('.l.')
                    league_id = league_part.split('.')[0] if '.' in league_part else league_part
                    return f"{game_id}.l.{league_id}.t.{team_key}"
            except Exception:
                pass
        
        # Return as-is if we can't normalize
        return team_key
    
    # Game-related tools
    def create_get_all_yahoo_fantasy_game_keys_tool(self):
        """Create a tool for retrieving all Yahoo Fantasy game keys."""
        @tool("get_all_yahoo_fantasy_game_keys", description="Get all Yahoo Fantasy Sports game keys by ID (from year of inception to present), sorted by season/year.")
        def get_all_yahoo_fantasy_game_keys() -> str:
            """Get all Yahoo Fantasy game keys."""
            try:
                data = self.query.get_all_yahoo_fantasy_game_keys()
                return f'All Yahoo Fantasy game keys: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch game keys: {e}"
        return get_all_yahoo_fantasy_game_keys
    
    def create_get_game_key_by_season_tool(self):
        """Create a tool for retrieving game key by season."""
        @tool("get_game_key_by_season", description="Get specific game key by season (e.g., '2023').")
        def get_game_key_by_season(season: str) -> str:
            """Get game key by season."""
            try:
                data = self.query.get_game_key_by_season(season)
                return f'Game key for season {season}: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch game key: {e}"
        return get_game_key_by_season
    
    def create_get_current_game_info_tool(self):
        """Create a tool for retrieving current game info."""
        @tool("get_current_game_info", description="Get game info for current fantasy season.")
        def get_current_game_info() -> str:
            """Get current game info."""
            try:
                data = self.query.get_current_game_info()
                return f'Current game info: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch current game info: {e}"
        return get_current_game_info
    
    def create_get_current_game_metadata_tool(self):
        """Create a tool for retrieving current game metadata."""
        @tool("get_current_game_metadata", description="Get game metadata for current fantasy season.")
        def get_current_game_metadata() -> str:
            """Get current game metadata."""
            try:
                data = self.query.get_current_game_metadata()
                return f'Current game metadata: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch current game metadata: {e}"
        return get_current_game_metadata
    
    def create_get_game_info_by_game_id_tool(self):
        """Create a tool for retrieving game info by game ID."""
        @tool("get_game_info_by_game_id", description="Get game info for specific game by ID.")
        def get_game_info_by_game_id(game_id: int) -> str:
            """Get game info by game ID."""
            try:
                data = self.query.get_game_info_by_game_id(game_id)
                return f'Game info for game ID {game_id}: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch game info: {e}"
        return get_game_info_by_game_id
    
    def create_get_game_metadata_by_game_id_tool(self):
        """Create a tool for retrieving game metadata by game ID."""
        @tool("get_game_metadata_by_game_id", description="Get game metadata for specific game by ID.")
        def get_game_metadata_by_game_id(game_id: int) -> str:
            """Get game metadata by game ID."""
            try:
                data = self.query.get_game_metadata_by_game_id(game_id)
                return f'Game metadata for game ID {game_id}: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch game metadata: {e}"
        return get_game_metadata_by_game_id
    
    def create_get_game_weeks_by_game_id_tool(self):
        """Create a tool for retrieving game weeks by game ID."""
        @tool("get_game_weeks_by_game_id", description="Get all valid weeks of a specific game by ID.")
        def get_game_weeks_by_game_id(game_id: int) -> str:
            """Get game weeks by game ID."""
            try:
                data = self.query.get_game_weeks_by_game_id(game_id)
                return f'Game weeks for game ID {game_id}: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch game weeks: {e}"
        return get_game_weeks_by_game_id
    
    def create_get_game_stat_categories_by_game_id_tool(self):
        """Create a tool for retrieving game stat categories by game ID."""
        @tool("get_game_stat_categories_by_game_id", description="Get all valid stat categories of a specific game by ID.")
        def get_game_stat_categories_by_game_id(game_id: int) -> str:
            """Get game stat categories by game ID."""
            try:
                data = self.query.get_game_stat_categories_by_game_id(game_id)
                return f'Stat categories for game ID {game_id}: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch stat categories: {e}"
        return get_game_stat_categories_by_game_id
    
    def create_get_game_position_types_by_game_id_tool(self):
        """Create a tool for retrieving game position types by game ID."""
        @tool("get_game_position_types_by_game_id", description="Get all valid position types for specific game by ID sorted alphabetically by type.")
        def get_game_position_types_by_game_id(game_id: int) -> str:
            """Get game position types by game ID."""
            try:
                data = self.query.get_game_position_types_by_game_id(game_id)
                return f'Position types for game ID {game_id}: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch position types: {e}"
        return get_game_position_types_by_game_id
    
    def create_get_game_roster_positions_by_game_id_tool(self):
        """Create a tool for retrieving game roster positions by game ID."""
        @tool("get_game_roster_positions_by_game_id", description="Get all valid roster positions for specific game by ID sorted alphabetically by position.")
        def get_game_roster_positions_by_game_id(game_id: int) -> str:
            """Get game roster positions by game ID."""
            try:
                data = self.query.get_game_roster_positions_by_game_id(game_id)
                return f'Roster positions for game ID {game_id}: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch roster positions: {e}"
        return get_game_roster_positions_by_game_id
    
    # User-related tools
    def create_get_current_user_tool(self):
        """Create a tool for retrieving current user info."""
        @tool("get_current_user", description="Get metadata for current logged-in user.")
        def get_current_user() -> str:
            """Get current user info."""
            try:
                data = self.query.get_current_user()
                return f'Current user info: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch current user: {e}"
        return get_current_user
    
    def create_get_user_games_tool(self):
        """Create a tool for retrieving user games."""
        @tool("get_user_games", description="Get game history for current logged-in user sorted by season/year.")
        def get_user_games() -> str:
            """Get user games."""
            try:
                data = self.query.get_user_games()
                return f'User games: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch user games: {e}"
        return get_user_games
    
    def create_get_user_leagues_by_game_key_tool(self):
        """Create a tool for retrieving user leagues by game key."""
        @tool("get_user_leagues_by_game_key", description="Get leagues for current logged-in user by game key.")
        def get_user_leagues_by_game_key(game_key: str) -> str:
            """Get user leagues by game key."""
            try:
                data = self.query.get_user_leagues_by_game_key(game_key)
                return f'User leagues for game key {game_key}: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch user leagues: {e}"
        return get_user_leagues_by_game_key
    
    def create_get_user_teams_tool(self):
        """Create a tool for retrieving user teams."""
        @tool("get_user_teams", description="Get teams for current logged-in user.")
        def get_user_teams() -> str:
            """Get user teams."""
            try:
                data = self.query.get_user_teams()
                return f'User teams: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch user teams: {e}"
        return get_user_teams
    
    # League-related tools
    def create_get_league_key_tool(self):
        """Create a tool for retrieving league key."""
        @tool("get_league_key", description="Get league key for selected league.")
        def get_league_key() -> str:
            """Get league key."""
            try:
                data = self.query.get_league_key()
                return f'League key: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch league key: {e}"
        return get_league_key
    
    def create_get_league_info_tool(self):
        """Create a tool for retrieving league info."""
        @tool("get_league_info", description="Get league info for selected league.")
        def get_league_info() -> str:
            """Get league info."""
            try:
                data = self.query.get_league_info()
                return f'League info: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch league info: {e}"
        return get_league_info
    
    def create_get_league_metadata_tool(self):
        """Create a tool for retrieving Yahoo League Metadata."""
        @tool("get_league_metadata", description="Get Yahoo League Metadata in json format from YPFS.")
        def get_league_metadata() -> str:
            """Get Yahoo League Metadata."""
            try:
                data = self.query.get_league_metadata()
                return f'League metadata: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch league metadata: {e}"
        return get_league_metadata
    
    def create_get_league_settings_tool(self):
        """Create a tool for retrieving league settings."""
        @tool("get_league_settings", description="Get league settings for selected league.")
        def get_league_settings() -> str:
            """Get league settings."""
            try:
                data = self.query.get_league_settings()
                return f'League settings: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch league settings: {e}"
        return get_league_settings
    
    def create_get_league_standings_tool(self):
        """Create a tool for retrieving league standings."""
        @tool("get_league_standings", description="Get league standings for selected league.")
        def get_league_standings() -> str:
            """Get league standings."""
            try:
                data = self.query.get_league_standings()
                return f'League standings: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch league standings: {e}"
        return get_league_standings
    
    def create_get_league_teams_tool(self):
        """Create a tool for retrieving league teams."""
        @tool("get_league_teams", description="Get all teams in the selected league. Returns team data including team_id and team_key (format: game_id.l.league_id.t.team_id). Team-related tools like get_team_info, get_team_metadata, etc. accept either team_id (numeric) or team_key.")
        def get_league_teams() -> str:
            """Get league teams. Returns team data including team_id and team_key that can be used for team-related API calls."""
            try:
                data = self.query.get_league_teams()
                return f'League teams: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch league teams: {e}"
        return get_league_teams
    
    def create_get_league_players_tool(self):
        """Create a tool for retrieving league players."""
        @tool("get_league_players", description="Get all players in the selected league.")
        def get_league_players() -> str:
            """Get league players."""
            try:
                data = self.query.get_league_players()
                return f'League players: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch league players: {e}"
        return get_league_players
    
    def create_get_league_draft_results_tool(self):
        """Create a tool for retrieving league draft results."""
        @tool("get_league_draft_results", description="Get draft results for selected league.")
        def get_league_draft_results() -> str:
            """Get league draft results."""
            try:
                data = self.query.get_league_draft_results()
                return f'League draft results: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch draft results: {e}"
        return get_league_draft_results
    
    def create_get_league_transactions_tool(self):
        """Create a tool for retrieving league transactions."""
        @tool("get_league_transactions", description="Get transactions for selected league.")
        def get_league_transactions() -> str:
            """Get league transactions."""
            try:
                data = self.query.get_league_transactions()
                return f'League transactions: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch transactions: {e}"
        return get_league_transactions
    
    def create_get_league_scoreboard_by_week_tool(self):
        """Create a tool for retrieving league scoreboard by week."""
        @tool("get_league_scoreboard_by_week", description="Get league scoreboard for a specific week.")
        def get_league_scoreboard_by_week(week: int) -> str:
            """Get league scoreboard by week."""
            try:
                data = self.query.get_league_scoreboard_by_week(week)
                return f'League scoreboard for week {week}: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch scoreboard: {e}"
        return get_league_scoreboard_by_week
    
    def create_get_league_matchups_by_week_tool(self):
        """Create a tool for retrieving league matchups by week."""
        @tool("get_league_matchups_by_week", description="Get league matchups for a specific week.")
        def get_league_matchups_by_week(week: int) -> str:
            """Get league matchups by week."""
            try:
                data = self.query.get_league_matchups_by_week(week)
                return f'League matchups for week {week}: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch matchups: {e}"
        return get_league_matchups_by_week
    
    # Team-related tools
    def create_get_team_info_tool(self):
        """Create a tool for retrieving team info."""
        @tool("get_team_info", description="Get team info by team_id or team key. Can accept team_id (e.g., 6) or team key format: {game_id}.l.{league_id}.t.{team_id} (e.g., 466.l.58930.t.6). Use get_league_teams first to get valid team IDs or keys.")
        def get_team_info(team_key_or_id: str) -> str:
            """Get team info. Accepts team_id (numeric) or team key format: game_id.l.league_id.t.team_id"""
            try:
                team_id = self._extract_team_id(team_key_or_id)
                data = self.query.get_team_info(team_id)
                return f'Team info for team_id {team_id}: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch team info for {team_key_or_id}: {e}. Make sure to use get_league_teams first to get valid team IDs or keys."
        return get_team_info
    
    def create_get_team_metadata_tool(self):
        """Create a tool for retrieving team metadata."""
        @tool("get_team_metadata", description="Get team metadata by team_id or team key. Can accept team_id (e.g., 6) or team key format: {game_id}.l.{league_id}.t.{team_id} (e.g., 466.l.58930.t.6). Use get_league_teams first to get valid team IDs or keys.")
        def get_team_metadata(team_key_or_id: str) -> str:
            """Get team metadata. Accepts team_id (numeric) or team key format: game_id.l.league_id.t.team_id"""
            try:
                team_id = self._extract_team_id(team_key_or_id)
                data = self.query.get_team_metadata(team_id)
                return f'Team metadata for team_id {team_id}: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch team metadata for {team_key_or_id}: {e}. Make sure to use get_league_teams first to get valid team IDs or keys."
        return get_team_metadata
    
    def create_get_team_stats_tool(self):
        """Create a tool for retrieving team stats."""
        @tool("get_team_stats", description="Get team stats by team_id or team key. Can accept team_id (e.g., 6) or team key format: {game_id}.l.{league_id}.t.{team_id} (e.g., 466.l.58930.t.6). Use get_league_teams first to get valid team IDs or keys.")
        def get_team_stats(team_key_or_id: str) -> str:
            """Get team stats. Accepts team_id (numeric) or team key format: game_id.l.league_id.t.team_id"""
            try:
                team_id = self._extract_team_id(team_key_or_id)
                data = self.query.get_team_stats(team_id)
                return f'Team stats for team_id {team_id}: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch team stats for {team_key_or_id}: {e}. Make sure to use get_league_teams first to get valid team IDs or keys."
        return get_team_stats
    
    def create_get_team_stats_by_week_tool(self):
        """Create a tool for retrieving team stats by week."""
        @tool("get_team_stats_by_week", description="Get team stats by team_id or team key and week. Can accept team_id (e.g., 6) or team key format: {game_id}.l.{league_id}.t.{team_id} (e.g., 466.l.58930.t.6). Week can be an integer or 'current' for the current week. Use get_league_teams first to get valid team IDs or keys.")
        def get_team_stats_by_week(team_key_or_id: str, week: Union[int, str] = "current") -> str:
            """Get team stats by week. Accepts team_id (numeric) or team key format: game_id.l.league_id.t.team_id. Week can be an integer or "current" for the current week."""
            try:
                team_id = self._extract_team_id(team_key_or_id)
                # Convert week to int if it's a numeric string, otherwise keep as-is (for "current")
                if isinstance(week, str) and week.lower() != "current":
                    week = int(week)
                elif isinstance(week, str) and week.lower() == "current":
                    week = "current"
                data = self.query.get_team_stats_by_week(team_id, week)
                return f'Team stats for team_id {team_id} in week {week}: {self._format_response(data)}'
            except KeyError as e:
                # Handle missing fields in API response (e.g., 'team_projected_points')
                # This tool only works with points leagues, not categories leagues
                return f"Failed to fetch team stats for team_id {team_key_or_id}: Missing field {e}. This tool only works with POINTS leagues, not categories leagues. The team_projected_points field is only available in points leagues. Make sure to use get_league_teams first to get valid team_id or team_key, and ensure you're using a points league."
            except Exception as e:
                return f"Failed to fetch team stats for team_id {team_key_or_id}: {e}. Make sure to use get_league_teams first to get valid team_id or team_key."
        return get_team_stats_by_week
    
    def create_get_team_standings_tool(self):
        """Create a tool for retrieving team standings."""
        @tool("get_team_standings", description="Get team standings by team_id or team key. Can accept team_id (e.g., 6) or team key format: {game_id}.l.{league_id}.t.{team_id} (e.g., 466.l.58930.t.6). Use get_league_teams first to get valid team IDs or keys.")
        def get_team_standings(team_key_or_id: str) -> str:
            """Get team standings. Accepts team_id (numeric) or team key format: game_id.l.league_id.t.team_id"""
            try:
                team_id = self._extract_team_id(team_key_or_id)
                data = self.query.get_team_standings(team_id)
                return f'Team standings for team_id {team_id}: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch team standings for {team_key_or_id}: {e}. Make sure to use get_league_teams first to get valid team IDs or keys."
        return get_team_standings
    
    def create_get_team_roster_by_week_tool(self):
        """Create a tool for retrieving team roster by week."""
        @tool("get_team_roster_by_week", description="Get team roster by team_id or team key and week. Can accept team_id (e.g., 6) or team key format: {game_id}.l.{league_id}.t.{team_id} (e.g., 466.l.58930.t.6). Use get_league_teams first to get valid team IDs or keys.")
        def get_team_roster_by_week(team_key_or_id: str, week: int) -> str:
            """Get team roster by week. Accepts team_id (numeric) or team key format: game_id.l.league_id.t.team_id"""
            try:
                team_id = self._extract_team_id(team_key_or_id)
                data = self.query.get_team_roster_by_week(team_id, week)
                return f'Team roster for team_id {team_id} in week {week}: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch team roster for {team_key_or_id}: {e}. Make sure to use get_league_teams first to get valid team IDs or keys."
        return get_team_roster_by_week
    
    def create_get_team_roster_player_info_by_week_tool(self):
        """Create a tool for retrieving team roster player info by week."""
        @tool("get_team_roster_player_info_by_week", description="Get team roster player info by team_id or team key and week. Can accept team_id (e.g., 6) or team key format: {game_id}.l.{league_id}.t.{team_id} (e.g., 466.l.58930.t.6). Use get_league_teams first to get valid team IDs or keys.")
        def get_team_roster_player_info_by_week(team_key_or_id: str, week: int) -> str:
            """Get team roster player info by week. Accepts team_id (numeric) or team key format: game_id.l.league_id.t.team_id"""
            try:
                team_id = self._extract_team_id(team_key_or_id)
                data = self.query.get_team_roster_player_info_by_week(team_id, week)
                return f'Team roster player info for team_id {team_id} in week {week}: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch roster player info for {team_key_or_id}: {e}. Make sure to use get_league_teams first to get valid team IDs or keys."
        return get_team_roster_player_info_by_week
    
    def create_get_team_roster_player_info_by_date_tool(self):
        """Create a tool for retrieving team roster player info by date."""
        @tool("get_team_roster_player_info_by_date", description="Get team roster player info by team_id or team key and date (YYYY-MM-DD format). Can accept team_id (e.g., 6) or team key format: {game_id}.l.{league_id}.t.{team_id} (e.g., 466.l.58930.t.6). Use get_league_teams first to get valid team IDs or keys.")
        def get_team_roster_player_info_by_date(team_key_or_id: str, date: str) -> str:
            """Get team roster player info by date. Accepts team_id (numeric) or team key format: game_id.l.league_id.t.team_id"""
            try:
                team_id = self._extract_team_id(team_key_or_id)
                data = self.query.get_team_roster_player_info_by_date(team_id, date)
                return f'Team roster player info for team_id {team_id} on {date}: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch roster player info for {team_key_or_id}: {e}. Make sure to use get_league_teams first to get valid team IDs or keys."
        return get_team_roster_player_info_by_date
    
    def create_get_team_roster_player_stats_tool(self):
        """Create a tool for retrieving team roster player stats."""
        @tool("get_team_roster_player_stats", description="Get team roster player stats by team_id or team key. Can accept team_id (e.g., 6) or team key format: {game_id}.l.{league_id}.t.{team_id} (e.g., 466.l.58930.t.6). Use get_league_teams first to get valid team IDs or keys.")
        def get_team_roster_player_stats(team_key_or_id: str) -> str:
            """Get team roster player stats. Accepts team_id (numeric) or team key format: game_id.l.league_id.t.team_id"""
            try:
                team_id = self._extract_team_id(team_key_or_id)
                data = self.query.get_team_roster_player_stats(team_id)
                return f'Team roster player stats for team_id {team_id}: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch roster player stats for {team_key_or_id}: {e}. Make sure to use get_league_teams first to get valid team IDs or keys."
        return get_team_roster_player_stats
    
    def create_get_team_roster_player_stats_by_week_tool(self):
        """Create a tool for retrieving team roster player stats by week."""
        @tool("get_team_roster_player_stats_by_week", description="Get team roster player stats by team_id or team key and week. Can accept team_id (e.g., 6) or team key format: {game_id}.l.{league_id}.t.{team_id} (e.g., 466.l.58930.t.6). Use get_league_teams first to get valid team IDs or keys.")
        def get_team_roster_player_stats_by_week(team_key_or_id: str, week: int) -> str:
            """Get team roster player stats by week. Accepts team_id (numeric) or team key format: game_id.l.league_id.t.team_id"""
            try:
                team_id = self._extract_team_id(team_key_or_id)
                data = self.query.get_team_roster_player_stats_by_week(team_id, week)
                return f'Team roster player stats for team_id {team_id} in week {week}: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch roster player stats for {team_key_or_id}: {e}. Make sure to use get_league_teams first to get valid team IDs or keys."
        return get_team_roster_player_stats_by_week
    
    def create_get_team_draft_results_tool(self):
        """Create a tool for retrieving team draft results."""
        @tool("get_team_draft_results", description="Get team draft results by team_id or team key. Can accept team_id (e.g., 6) or team key format: {game_id}.l.{league_id}.t.{team_id} (e.g., 466.l.58930.t.6). Use get_league_teams first to get valid team IDs or keys.")
        def get_team_draft_results(team_key_or_id: str) -> str:
            """Get team draft results. Accepts team_id (numeric) or team key format: game_id.l.league_id.t.team_id"""
            try:
                team_id = self._extract_team_id(team_key_or_id)
                data = self.query.get_team_draft_results(team_id)
                return f'Team draft results for team_id {team_id}: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch team draft results for {team_key_or_id}: {e}. Make sure to use get_league_teams first to get valid team IDs or keys."
        return get_team_draft_results
    
    def create_get_team_matchups_tool(self):
        """Create a tool for retrieving team matchups."""
        @tool("get_team_matchups", description="Get team matchups by team_id or team key. Can accept team_id (e.g., 6) or team key format: {game_id}.l.{league_id}.t.{team_id} (e.g., 466.l.58930.t.6). Use get_league_teams first to get valid team IDs or keys.")
        def get_team_matchups(team_key_or_id: str) -> str:
            """Get team matchups. Accepts team_id (numeric) or team key format: game_id.l.league_id.t.team_id"""
            try:
                team_id = self._extract_team_id(team_key_or_id)
                data = self.query.get_team_matchups(team_id)
                return f'Team matchups for team_id {team_id}: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch team matchups for {team_key_or_id}: {e}. Make sure to use get_league_teams first to get valid team IDs or keys."
        return get_team_matchups
    
    # Player-related tools
    def create_get_player_stats_for_season_tool(self):
        """Create a tool for retrieving player stats for season."""
        @tool("get_player_stats_for_season", description="Get player stats for season by player key.")
        def get_player_stats_for_season(player_key: str) -> str:
            """Get player stats for season."""
            try:
                data = self.query.get_player_stats_for_season(player_key)
                return f'Player stats for season for {player_key}: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch player stats: {e}"
        return get_player_stats_for_season
    
    def create_get_player_stats_by_week_tool(self):
        """Create a tool for retrieving player stats by week."""
        @tool("get_player_stats_by_week", description="Get player stats by week for player key.")
        def get_player_stats_by_week(player_key: str, week: int) -> str:
            """Get player stats by week."""
            try:
                data = self.query.get_player_stats_by_week(player_key, week)
                return f'Player stats for {player_key} in week {week}: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch player stats: {e}"
        return get_player_stats_by_week
    
    def create_get_player_stats_by_date_tool(self):
        """Create a tool for retrieving player stats by date."""
        @tool("get_player_stats_by_date", description="Get player stats by date (YYYY-MM-DD format) for player key.")
        def get_player_stats_by_date(player_key: str, date: str) -> str:
            """Get player stats by date."""
            try:
                data = self.query.get_player_stats_by_date(player_key, date)
                return f'Player stats for {player_key} on {date}: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch player stats: {e}"
        return get_player_stats_by_date
    
    def create_get_player_ownership_tool(self):
        """Create a tool for retrieving player ownership."""
        @tool("get_player_ownership", description="Get player ownership by player key.")
        def get_player_ownership(player_key: str) -> str:
            """Get player ownership."""
            try:
                data = self.query.get_player_ownership(player_key)
                return f'Player ownership for {player_key}: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch player ownership: {e}"
        return get_player_ownership
    
    def create_get_player_percent_owned_by_week_tool(self):
        """Create a tool for retrieving player percent owned by week."""
        @tool("get_player_percent_owned_by_week", description="Get player percent owned by week for player key.")
        def get_player_percent_owned_by_week(player_key: str, week: int) -> str:
            """Get player percent owned by week."""
            try:
                data = self.query.get_player_percent_owned_by_week(player_key, week)
                return f'Player percent owned for {player_key} in week {week}: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch player percent owned: {e}"
        return get_player_percent_owned_by_week
    
    def create_get_player_draft_analysis_tool(self):
        """Create a tool for retrieving player draft analysis."""
        @tool("get_player_draft_analysis", description="Get draft analysis of specific player by player key.")
        def get_player_draft_analysis(player_key: str) -> str:
            """Get player draft analysis."""
            try:
                data = self.query.get_player_draft_analysis(player_key)
                return f'Player draft analysis for {player_key}: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch player draft analysis: {e}"
        return get_player_draft_analysis
    
    def get_all_tools(self):
        """
        Get all available Yahoo Fantasy tools.
        
        Returns:
            List of all LangChain tool instances
        """
        tools = [
            # Game tools
            self.create_get_all_yahoo_fantasy_game_keys_tool(),
            self.create_get_game_key_by_season_tool(),
            self.create_get_current_game_info_tool(),
            self.create_get_current_game_metadata_tool(),
            self.create_get_game_info_by_game_id_tool(),
            self.create_get_game_metadata_by_game_id_tool(),
            self.create_get_game_weeks_by_game_id_tool(),
            self.create_get_game_stat_categories_by_game_id_tool(),
            self.create_get_game_position_types_by_game_id_tool(),
            self.create_get_game_roster_positions_by_game_id_tool(),
            # User tools
            self.create_get_current_user_tool(),
            self.create_get_user_games_tool(),
            self.create_get_user_leagues_by_game_key_tool(),
            self.create_get_user_teams_tool(),
            # League tools
            self.create_get_league_key_tool(),
            self.create_get_league_info_tool(),
            self.create_get_league_metadata_tool(),
            self.create_get_league_settings_tool(),
            self.create_get_league_standings_tool(),
            self.create_get_league_teams_tool(),
            self.create_get_league_players_tool(),
            self.create_get_league_draft_results_tool(),
            self.create_get_league_transactions_tool(),
            self.create_get_league_scoreboard_by_week_tool(),
            self.create_get_league_matchups_by_week_tool(),
            # Team tools
            self.create_get_team_info_tool(),
            self.create_get_team_metadata_tool(),
            self.create_get_team_stats_tool(),
            self.create_get_team_stats_by_week_tool(),
            self.create_get_team_standings_tool(),
            self.create_get_team_roster_by_week_tool(),
            self.create_get_team_roster_player_info_by_week_tool(),
            self.create_get_team_roster_player_info_by_date_tool(),
            self.create_get_team_roster_player_stats_tool(),
            self.create_get_team_roster_player_stats_by_week_tool(),
            self.create_get_team_draft_results_tool(),
            self.create_get_team_matchups_tool(),
            # Player tools
            self.create_get_player_stats_for_season_tool(),
            self.create_get_player_stats_by_week_tool(),
            self.create_get_player_stats_by_date_tool(),
            self.create_get_player_ownership_tool(),
            self.create_get_player_percent_owned_by_week_tool(),
            self.create_get_player_draft_analysis_tool(),
        ]
        return tools
