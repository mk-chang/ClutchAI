"""
NBA API tools for ClutchAI Agent.
"""

import json
from typing import Optional
from langchain_core.tools import tool

# NBA API imports
# Available endpoints: https://github.com/swar/nba_api/tree/master/docs/nba_api/stats/endpoints
from nba_api.stats.endpoints import (
    playercareerstats,
    playergamelog,
    playerdashboardbygamesplits,
    playerdashboardbygeneralsplits,
    teamgamelog,
    scoreboardv2,
    boxscoretraditionalv2,
    playbyplayv2,
    commonplayerinfo,
    commonteamroster,
    leaguegamefinder,
)
from nba_api.live.nba.endpoints import scoreboard as live_scoreboard
from nba_api.stats.static import players, teams
from .base import ClutchAITool


class nbaAPITool(ClutchAITool):
    """
    Class for creating NBA API tools for LangChain agents.
    Provides tools for accessing NBA.com stats and live data.
    """
    
    def __init__(self, timeout: int = 30, debug: bool = False):
        """
        Initialize nbaAPITool.
        
        Args:
            timeout: Request timeout in seconds (default: 30)
            debug: Enable debug logging (default: False)
        """
        super().__init__(debug=debug)
        self.timeout = timeout
    
    def _format_response(self, data) -> str:
        """Helper method to format API response data as JSON string."""
        try:
            # Handle pandas DataFrames
            if hasattr(data, 'to_dict'):
                return json.dumps(data.to_dict('records'), default=str, indent=2)
            # Handle nba_api response objects with get_dict() method
            elif hasattr(data, 'get_dict'):
                return json.dumps(data.get_dict(), default=str, indent=2)
            # Handle nba_api response objects with get_json() method
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
    
    def _format_dataframe_response(self, data) -> str:
        """Helper method to format pandas DataFrame response."""
        try:
            if hasattr(data, 'to_dict'):
                return json.dumps(data.to_dict('records'), default=str, indent=2)
            elif hasattr(data, 'get_data_frame'):
                df = data.get_data_frame()
                return json.dumps(df.to_dict('records'), default=str, indent=2)
            else:
                return self._format_response(data)
        except Exception as e:
            return f"Error formatting DataFrame: {str(e)}\nRaw data: {str(data)}"
    
    # Static Data Tools
    def create_get_all_players_tool(self):
        """Create a tool for retrieving all NBA players."""
        @tool("get_all_nba_players", description="Get all NBA players from static data. Returns player information including player_id, full_name, and other details.")
        def get_all_nba_players() -> str:
            """Get all NBA players."""
            try:
                data = players.get_players()
                return f'All NBA players: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch players: {e}"
        return get_all_nba_players
    
    def create_find_players_by_name_tool(self):
        """Create a tool for finding players by name."""
        @tool("find_nba_players_by_name", description="Find NBA players by full name or partial name. Returns matching players with their player_id.")
        def find_nba_players_by_name(full_name: str) -> str:
            """Find players by name."""
            try:
                data = players.find_players_by_full_name(full_name)
                return f'Players matching "{full_name}": {self._format_response(data)}'
            except Exception as e:
                return f"Failed to find players: {e}"
        return find_nba_players_by_name
    
    def create_get_all_teams_tool(self):
        """Create a tool for retrieving all NBA teams."""
        @tool("get_all_nba_teams", description="Get all NBA teams from static data. Returns team information including team_id, full_name, abbreviation, and other details.")
        def get_all_nba_teams() -> str:
            """Get all NBA teams."""
            try:
                data = teams.get_teams()
                return f'All NBA teams: {self._format_response(data)}'
            except Exception as e:
                return f"Failed to fetch teams: {e}"
        return get_all_nba_teams
    
    def create_find_team_by_name_tool(self):
        """Create a tool for finding teams by name."""
        @tool("find_nba_team_by_name", description="Find NBA team by full name or abbreviation. Returns team information with team_id.")
        def find_nba_team_by_name(full_name: str) -> str:
            """Find team by name."""
            try:
                data = teams.find_teams_by_full_name(full_name)
                if not data:
                    # Try abbreviation
                    data = teams.find_teams_by_abbreviation(full_name)
                return f'Team matching "{full_name}": {self._format_response(data)}'
            except Exception as e:
                return f"Failed to find team: {e}"
        return find_nba_team_by_name
    
    # Player Stats Tools
    def create_get_player_info_tool(self):
        """Create a tool for retrieving player information."""
        @tool("get_player_info", description="Get detailed information about an NBA player by player_id. Use find_nba_players_by_name first to get player_id.")
        def get_player_info(player_id: str) -> str:
            """Get player information."""
            try:
                data = commonplayerinfo.CommonPlayerInfo(player_id=player_id, timeout=self.timeout)
                return f'Player info for player_id {player_id}: {self._format_response(data.get_dict())}'
            except Exception as e:
                return f"Failed to fetch player info: {e}"
        return get_player_info
    
    def create_get_player_career_stats_tool(self):
        """Create a tool for retrieving player career stats."""
        @tool("get_player_career_stats", description="Get career statistics for an NBA player by player_id. Returns regular season and playoff stats. Use find_nba_players_by_name first to get player_id.")
        def get_player_career_stats(player_id: str) -> str:
            """Get player career stats."""
            try:
                data = playercareerstats.PlayerCareerStats(player_id=player_id, timeout=self.timeout)
                # Get both regular season and playoff data
                result = {
                    "regular_season_totals": data.season_totals_regular_season.get_data_frame().to_dict('records') if hasattr(data, 'season_totals_regular_season') else None,
                    "playoff_totals": data.season_totals_post_season.get_data_frame().to_dict('records') if hasattr(data, 'season_totals_post_season') else None,
                }
                return f'Career stats for player_id {player_id}: {json.dumps(result, default=str, indent=2)}'
            except Exception as e:
                return f"Failed to fetch career stats: {e}"
        return get_player_career_stats
    
    def create_get_player_game_log_tool(self):
        """Create a tool for retrieving player game log."""
        @tool("get_player_game_log", description="Get game log for an NBA player by player_id and season (e.g., '2023-24'). Use find_nba_players_by_name first to get player_id.")
        def get_player_game_log(player_id: str, season: str = "2023-24") -> str:
            """Get player game log."""
            try:
                data = playergamelog.PlayerGameLog(player_id=player_id, season=season, timeout=self.timeout)
                df = data.get_data_frame()
                return f'Game log for player_id {player_id} in season {season}: {json.dumps(df.to_dict("records"), default=str, indent=2)}'
            except Exception as e:
                return f"Failed to fetch game log: {e}"
        return get_player_game_log
    
    def create_get_player_dashboard_by_game_splits_tool(self):
        """Create a tool for retrieving player dashboard by game splits."""
        @tool("get_player_dashboard_by_game_splits", description="Get player dashboard statistics broken down by game splits (home/away, pre/post all-star, etc.) by player_id and season. Use find_nba_players_by_name first to get player_id.")
        def get_player_dashboard_by_game_splits(player_id: str, season: str = "2023-24") -> str:
            """Get player dashboard by game splits."""
            try:
                data = playerdashboardbygamesplits.PlayerDashboardByGameSplits(
                    player_id=player_id, 
                    season=season, 
                    timeout=self.timeout
                )
                return f'Player dashboard by game splits for player_id {player_id}: {self._format_response(data.get_dict())}'
            except Exception as e:
                return f"Failed to fetch player dashboard: {e}"
        return get_player_dashboard_by_game_splits
    
    def create_get_player_dashboard_by_general_splits_tool(self):
        """Create a tool for retrieving player dashboard by general splits."""
        @tool("get_player_dashboard_by_general_splits", description="Get player dashboard statistics broken down by general splits (wins/losses, etc.) by player_id and season. Use find_nba_players_by_name first to get player_id.")
        def get_player_dashboard_by_general_splits(player_id: str, season: str = "2023-24") -> str:
            """Get player dashboard by general splits."""
            try:
                data = playerdashboardbygeneralsplits.PlayerDashboardByGeneralSplits(
                    player_id=player_id, 
                    season=season, 
                    timeout=self.timeout
                )
                return f'Player dashboard by general splits for player_id {player_id}: {self._format_response(data.get_dict())}'
            except Exception as e:
                return f"Failed to fetch player dashboard: {e}"
        return get_player_dashboard_by_general_splits
    
    # Team Stats Tools
    def create_get_team_info_tool(self):
        """Create a tool for retrieving team information."""
        @tool("get_team_info", description="Get detailed information about an NBA team by team_id. Use find_nba_team_by_name first to get team_id.")
        def get_team_info(team_id: str) -> str:
            """Get team information."""
            try:
                data = commonteamroster.CommonTeamRoster(team_id=team_id, season="2023-24", timeout=self.timeout)
                return f'Team info for team_id {team_id}: {self._format_response(data.get_dict())}'
            except Exception as e:
                return f"Failed to fetch team info: {e}"
        return get_team_info
    
    def create_get_team_game_log_tool(self):
        """Create a tool for retrieving team game log."""
        @tool("get_team_game_log", description="Get game log for an NBA team by team_id and season (e.g., '2023-24'). Use find_nba_team_by_name first to get team_id.")
        def get_team_game_log(team_id: str, season: str = "2023-24") -> str:
            """Get team game log."""
            try:
                data = teamgamelog.TeamGameLog(team_id=team_id, season=season, timeout=self.timeout)
                df = data.get_data_frame()
                return f'Game log for team_id {team_id} in season {season}: {json.dumps(df.to_dict("records"), default=str, indent=2)}'
            except Exception as e:
                return f"Failed to fetch team game log: {e}"
        return get_team_game_log
    
    # Game Tools
    def create_get_scoreboard_tool(self):
        """Create a tool for retrieving today's scoreboard."""
        @tool("get_nba_scoreboard", description="Get today's NBA scoreboard with game information and scores.")
        def get_nba_scoreboard(game_date: Optional[str] = None) -> str:
            """Get NBA scoreboard. If game_date is not provided, returns today's games. Format: YYYY-MM-DD."""
            try:
                if game_date:
                    data = scoreboardv2.ScoreboardV2(game_date=game_date, timeout=self.timeout)
                else:
                    data = scoreboardv2.ScoreboardV2(timeout=self.timeout)
                return f'NBA Scoreboard: {self._format_response(data.get_dict())}'
            except Exception as e:
                return f"Failed to fetch scoreboard: {e}"
        return get_nba_scoreboard
    
    def create_get_live_scoreboard_tool(self):
        """Create a tool for retrieving live scoreboard."""
        @tool("get_nba_live_scoreboard", description="Get live NBA scoreboard with real-time game data and scores.")
        def get_nba_live_scoreboard() -> str:
            """Get live NBA scoreboard."""
            try:
                data = live_scoreboard.ScoreBoard()
                return f'Live NBA Scoreboard: {self._format_response(data.get_dict())}'
            except Exception as e:
                return f"Failed to fetch live scoreboard: {e}"
        return get_nba_live_scoreboard
    
    def create_get_box_score_tool(self):
        """Create a tool for retrieving box score."""
        @tool("get_nba_box_score", description="Get box score for a specific game by game_id. Use get_nba_scoreboard first to get game_id.")
        def get_nba_box_score(game_id: str) -> str:
            """Get box score for a game."""
            try:
                data = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id, timeout=self.timeout)
                return f'Box score for game_id {game_id}: {self._format_response(data.get_dict())}'
            except Exception as e:
                return f"Failed to fetch box score: {e}"
        return get_nba_box_score
    
    def create_get_play_by_play_tool(self):
        """Create a tool for retrieving play-by-play data."""
        @tool("get_nba_play_by_play", description="Get play-by-play data for a specific game by game_id. Use get_nba_scoreboard first to get game_id.")
        def get_nba_play_by_play(game_id: str) -> str:
            """Get play-by-play data for a game."""
            try:
                data = playbyplayv2.PlayByPlayV2(game_id=game_id, timeout=self.timeout)
                return f'Play-by-play for game_id {game_id}: {self._format_response(data.get_dict())}'
            except Exception as e:
                return f"Failed to fetch play-by-play: {e}"
        return get_nba_play_by_play
    
    def create_find_games_tool(self):
        """Create a tool for finding games."""
        @tool("find_nba_games", description="Find NBA games by various criteria. Can filter by team_id, player_id, season, date_from, date_to, etc.")
        def find_nba_games(
            team_id: Optional[str] = None,
            player_id: Optional[str] = None,
            season: Optional[str] = None,
            date_from: Optional[str] = None,
            date_to: Optional[str] = None
        ) -> str:
            """Find NBA games by criteria."""
            try:
                # Build league_game_finder parameters
                params = {}
                if team_id:
                    params['team_id_nullable'] = team_id
                if player_id:
                    params['player_id_nullable'] = player_id
                if season:
                    params['season_nullable'] = season
                if date_from:
                    params['date_from_nullable'] = date_from
                if date_to:
                    params['date_to_nullable'] = date_to
                
                data = leaguegamefinder.LeagueGameFinder(**params, timeout=self.timeout)
                df = data.get_data_frame()
                return f'Found games: {json.dumps(df.to_dict("records"), default=str, indent=2)}'
            except Exception as e:
                return f"Failed to find games: {e}"
        return find_nba_games
    
    def get_all_tools(self):
        """
        Get all available NBA API tools.
        
        Returns:
            List of all LangChain tool instances
        """
        tools = [
            # Static Data Tools
            self.create_get_all_players_tool(),
            self.create_find_players_by_name_tool(),
            self.create_get_all_teams_tool(),
            self.create_find_team_by_name_tool(),
            # Player Stats Tools
            self.create_get_player_info_tool(),
            self.create_get_player_career_stats_tool(),
            self.create_get_player_game_log_tool(),
            self.create_get_player_dashboard_by_game_splits_tool(),
            self.create_get_player_dashboard_by_general_splits_tool(),
            # Team Stats Tools
            self.create_get_team_info_tool(),
            self.create_get_team_game_log_tool(),
            # Game Tools
            self.create_get_scoreboard_tool(),
            self.create_get_live_scoreboard_tool(),
            self.create_get_box_score_tool(),
            self.create_get_play_by_play_tool(),
            self.create_find_games_tool(),
        ]
        return tools

