"""
User Context Gathering Module

This module provides functionality to gather user context about teams, leagues, and rosters
from the Yahoo Fantasy Sports API.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from yfpy.query import YahooFantasySportsQuery

from logger import get_logger

logger = get_logger(__name__)


class UserContextGatherer:
    """
    Class responsible for gathering user context from Yahoo Fantasy Sports API.
    
    This class encapsulates all logic for fetching and formatting user-specific
    context including team info, league info, standings, and roster data.
    """
    
    def __init__(
        self,
        query: Optional[YahooFantasySportsQuery] = None,
        team_name: Optional[str] = None
    ):
        """
        Initialize the UserContextGatherer.
        
        Args:
            query: YahooFantasySportsQuery instance for API access (None if Yahoo OAuth unavailable, e.g. headless)
            team_name: Name of the user's team (optional)
        """
        self.query = query
        self.team_name = team_name
        self.context_parts: List[str] = []
        self.matching_team_id: Optional[int] = None
        self.matching_team_key: Optional[str] = None
        self.current_week: Optional[str] = None
        # Display attributes for UI (populated during gather())
        self.display_current_date: Optional[str] = None
        self.display_league_name: Optional[str] = None
        self.display_league_size: Optional[int] = None
        self.display_rank: Optional[str] = None
        self.display_record: Optional[str] = None
        self.display_roster_preview: List[str] = []
    
    def gather(self) -> str:
        """
        Main method to gather all user context.
        
        Returns:
            Formatted context string with current date, user teams, leagues, rosters, and standings
        """
        try:
            self.context_parts = []
            self.display_roster_preview = []
            
            # Gather each section of context
            self._add_current_date_time()
            
            if self.query is None:
                self.context_parts.append("=== YAHOO FANTASY ===")
                self.context_parts.append(
                    "Yahoo Fantasy: Not available (OAuth not completed for this environment). "
                    "Complete OAuth once with this app's redirect URI and add token secrets to run with Yahoo data."
                )
                self.context_parts.append("")
            elif self.team_name:
                if self._find_matching_team():
                    self._add_team_info()
                    self._add_league_info()
                    self._add_team_standings()
                    self._add_current_roster()
                else:
                    self._add_team_not_found_message()
            
            context_str = "\n".join(self.context_parts)
            logger.info(f"Gathered user context ({len(context_str)} characters)")
            return context_str
            
        except Exception as e:
            logger.error(f"Error gathering user context: {e}")
            return f"Error gathering user context: {e}"
    
    def get_display_info(self) -> Dict[str, Any]:
        """
        Return structured display info for UI rendering.
        
        Must be called after gather(). Returns a dict with keys suitable for
        Streamlit display: current_date, team_name, league_name, league_size,
        current_week, rank, record, roster_preview.
        
        Returns:
            Dict with display values (None or empty list for missing data)
        """
        return {
            "current_date": self.display_current_date,
            "team_name": self.team_name,
            "league_name": self.display_league_name,
            "league_size": self.display_league_size,
            "current_week": self.current_week,
            "rank": self.display_rank,
            "record": self.display_record,
            "roster_preview": list(self.display_roster_preview),
        }
    
    def _add_current_date_time(self) -> None:
        """Add current date/time to context."""
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.display_current_date = current_date
        self.context_parts.append("=== CURRENT DATE/TIME ===")
        self.context_parts.append(f"Current Date/Time: {current_date}")
        self.context_parts.append("")
    
    def _find_matching_team(self) -> bool:
        """
        Find the team matching the team_name in the league.
        
        Returns:
            True if team was found, False otherwise
        """
        try:
            league_teams = self.query.get_league_teams()
            
            if not league_teams:
                return False
            
            # Process teams list
            teams_list = []
            if isinstance(league_teams, list):
                teams_list = league_teams
            elif hasattr(league_teams, 'team'):
                teams_list = league_teams.team if isinstance(league_teams.team, list) else [league_teams.team]
            
            # Search for team by name (case-insensitive)
            for team in teams_list:
                team_name_attr = self._extract_team_name(team)
                
                if team_name_attr and team_name_attr.lower().strip() == str(self.team_name).lower().strip():
                    # Found matching team
                    self.matching_team_id = self._extract_team_id(team)
                    self.matching_team_key = self._extract_team_key(team)
                    return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Could not fetch team info for '{self.team_name}': {e}")
            return False
    
    def _extract_team_name(self, team: Any) -> Optional[str]:
        """
        Extract team name from team object using multiple fallback methods.
        
        Args:
            team: Team object from Yahoo API
            
        Returns:
            Team name as string, or None if not found
        """
        team_name_attr = None
        
        # Try multiple ways to access the name attribute
        try:
            team_name_attr = team.name
        except (AttributeError, KeyError, TypeError):
            try:
                team_name_attr = getattr(team, 'name', None)
            except (AttributeError, KeyError, TypeError):
                pass
        
        # Try team_name attribute if name didn't work
        if not team_name_attr:
            try:
                team_name_attr = getattr(team, 'team_name', None)
            except (AttributeError, KeyError, TypeError):
                pass
        
        # Try dictionary-style access
        if not team_name_attr:
            try:
                if hasattr(team, '__getitem__'):
                    team_name_attr = team['name']
            except (TypeError, KeyError, AttributeError):
                pass
        
        # Try accessing via __dict__
        if not team_name_attr:
            try:
                if hasattr(team, '__dict__') and 'name' in team.__dict__:
                    team_name_attr = team.__dict__['name']
            except (AttributeError, KeyError):
                pass
        
        # Convert bytes to string if needed
        if team_name_attr and isinstance(team_name_attr, bytes):
            try:
                team_name_attr = team_name_attr.decode('utf-8')
            except (UnicodeDecodeError, AttributeError):
                team_name_attr = str(team_name_attr)
        
        # Ensure it's a string
        if team_name_attr:
            team_name_attr = str(team_name_attr)
        
        return team_name_attr
    
    def _extract_team_id(self, team: Any) -> Optional[int]:
        """
        Extract team ID from team object.
        
        Args:
            team: Team object from Yahoo API
            
        Returns:
            Team ID as int, or None if not found
        """
        if hasattr(team, 'team_id'):
            return getattr(team, 'team_id', None)
        elif hasattr(team, 'id'):
            return getattr(team, 'id', None)
        return None
    
    def _extract_team_key(self, team: Any) -> Optional[str]:
        """
        Extract team key from team object.
        
        Args:
            team: Team object from Yahoo API
            
        Returns:
            Team key as string, or None if not found
        """
        if hasattr(team, 'team_key'):
            return getattr(team, 'team_key', None)
        elif hasattr(team, 'key'):
            return getattr(team, 'key', None)
        return None
    
    def _add_team_info(self) -> None:
        """Add basic team information to context."""
        self.context_parts.append("=== USER TEAM INFO ===")
        self.context_parts.append(f"Team Name: {self.team_name}")
        if self.matching_team_id:
            self.context_parts.append(f"Team ID: {self.matching_team_id}")
        if self.matching_team_key:
            self.context_parts.append(f"Team Key: {self.matching_team_key}")
        self.context_parts.append("")
    
    def _add_league_info(self) -> None:
        """Add league information to context."""
        self.context_parts.append("=== LEAGUE INFO ===")
        
        # Get league size
        try:
            league_teams = self.query.get_league_teams()
            teams_list = []
            if isinstance(league_teams, list):
                teams_list = league_teams
            elif hasattr(league_teams, 'team'):
                teams_list = league_teams.team if isinstance(league_teams.team, list) else [league_teams.team]
            
            league_size = len(teams_list) if teams_list else 0
            self.display_league_size = league_size
            self.context_parts.append(f"League Size: {league_size} teams")
        except Exception as e:
            logger.debug(f"Could not fetch league size: {e}")
            self.context_parts.append("League Size: Not available")
        
        # Get league scoring type
        self._add_league_scoring_type()
        
        # Get current week
        self._get_current_week()
        
        # Get league metadata
        self._add_league_metadata()
        
        self.context_parts.append("")
    
    def _add_league_scoring_type(self) -> None:
        """Add league scoring type to context."""
        try:
            league_teams = self.query.get_league_teams()
            teams_list = []
            if isinstance(league_teams, list):
                teams_list = league_teams
            elif hasattr(league_teams, 'team'):
                teams_list = league_teams.team if isinstance(league_teams.team, list) else [league_teams.team]
            
            # Find matching team to get scoring type
            for team in teams_list:
                team_name_attr = self._extract_team_name(team)
                if team_name_attr and team_name_attr.lower().strip() == str(self.team_name).lower().strip():
                    league_scoring_type = getattr(team, 'league_scoring_type', None)
                    if league_scoring_type:
                        if isinstance(league_scoring_type, bytes):
                            league_scoring_type = league_scoring_type.decode('utf-8')
                        self.context_parts.append(f"League Scoring Type: {league_scoring_type}")
                    break
        except Exception:
            pass
    
    def _get_current_week(self) -> None:
        """Get and store the current week, and add it to context."""
        try:
            # First try to get current week from game info
            game_info = self.query.get_current_game_info()
            if game_info:
                # Try to get current_week directly from game_info
                self.current_week = getattr(game_info, 'current_week', None) or getattr(game_info, 'week', None)
                
                # If not available, calculate from current date using game_weeks
                if not self.current_week:
                    self.current_week = self._calculate_week_from_game_info(game_info)
            
            if self.current_week:
                self.context_parts.append(f"Current Week: {self.current_week}")
            else:
                self.context_parts.append("Current Week: Not available (off-season or season not started)")
        except Exception as e:
            logger.debug(f"Could not fetch current week: {e}")
            self.context_parts.append("Current Week: Not available (error fetching)")
    
    def _calculate_week_from_game_info(self, game_info: Any) -> Optional[str]:
        """
        Calculate the current week number from game_info using game_weeks.
        
        Args:
            game_info: Game object from get_current_game_info()
            
        Returns:
            Week number as string (1-25), or None if not found
        """
        try:
            # Get game_weeks from game_info
            game_weeks = None
            if hasattr(game_info, 'game_weeks'):
                game_weeks = getattr(game_info, 'game_weeks', None)
            
            if not game_weeks:
                return None
            
            # Get current date
            current_date = datetime.now().date()
            
            # Process game_weeks to find which week contains the current date
            weeks_list = []
            if isinstance(game_weeks, list):
                weeks_list = game_weeks
            elif hasattr(game_weeks, 'week'):
                weeks_list = game_weeks.week if isinstance(game_weeks.week, list) else [game_weeks.week]
            elif hasattr(game_weeks, 'weeks'):
                weeks_list = game_weeks.weeks if isinstance(game_weeks.weeks, list) else [game_weeks.weeks]
            
            # Find the week that contains the current date
            for week in weeks_list:
                try:
                    # Extract week object (could be wrapped in game_week dict)
                    week_obj = week
                    if isinstance(week, dict) and 'game_week' in week:
                        week_obj = week['game_week']
                    elif hasattr(week, 'game_week'):
                        week_obj = getattr(week, 'game_week', week)
                    
                    # Extract week number
                    week_num = None
                    if hasattr(week_obj, 'week'):
                        week_num = getattr(week_obj, 'week', None)
                    elif isinstance(week_obj, dict):
                        week_num = week_obj.get('week')
                    
                    # Extract start and end dates
                    start_date = None
                    end_date = None
                    
                    # Try to get start date
                    if hasattr(week_obj, 'start'):
                        start_attr = getattr(week_obj, 'start', None)
                        if start_attr:
                            if isinstance(start_attr, str):
                                start_date = datetime.strptime(start_attr.split('T')[0], "%Y-%m-%d").date()
                            elif hasattr(start_attr, 'date'):
                                start_date = start_attr.date()
                    
                    # Try to get end date
                    if hasattr(week_obj, 'end'):
                        end_attr = getattr(week_obj, 'end', None)
                        if end_attr:
                            if isinstance(end_attr, str):
                                end_date = datetime.strptime(end_attr.split('T')[0], "%Y-%m-%d").date()
                            elif hasattr(end_attr, 'date'):
                                end_date = end_attr.date()
                    
                    # Try dictionary access
                    if not start_date and isinstance(week_obj, dict):
                        start_str = week_obj.get('start')
                        if start_str:
                            start_date = datetime.strptime(str(start_str).split('T')[0], "%Y-%m-%d").date()
                    
                    if not end_date and isinstance(week_obj, dict):
                        end_str = week_obj.get('end')
                        if end_str:
                            end_date = datetime.strptime(str(end_str).split('T')[0], "%Y-%m-%d").date()
                    
                    # Check if current date falls within this week's range
                    if start_date and end_date and start_date <= current_date <= end_date:
                        if week_num:
                            return str(week_num)
                
                except Exception as e:
                    logger.debug(f"Error processing week in _calculate_week_from_game_info: {e}")
                    continue
            
            return None
            
        except Exception as e:
            logger.debug(f"Could not calculate week from game_info: {e}")
            return None
    
    def _add_league_metadata(self) -> None:
        """Add league metadata to context."""
        # Try get_league_info first
        try:
            league_info = self.query.get_league_info()
            if league_info:
                self.context_parts.append("League Metadata:")
                self._extract_league_info_fields(league_info)
        except Exception as e:
            logger.debug(f"Could not fetch league info: {e}")
        
        # Try get_league_metadata as well
        try:
            league_metadata = self.query.get_league_metadata()
            if league_metadata:
                # Add any additional metadata fields not already captured
                if hasattr(league_metadata, 'name') and not any('League Name' in part for part in self.context_parts):
                    league_name = getattr(league_metadata, 'name', None)
                    if league_name:
                        if isinstance(league_name, bytes):
                            league_name = league_name.decode('utf-8')
                        if not self.display_league_name:
                            self.display_league_name = str(league_name)
                        self.context_parts.append(f"  League Name: {league_name}")
        except Exception as e:
            logger.debug(f"Could not fetch league metadata: {e}")
    
    def _extract_league_info_fields(self, league_info: Any) -> None:
        """
        Extract and add league info fields to context.
        
        Args:
            league_info: League info object from Yahoo API
        """
        if hasattr(league_info, 'name'):
            league_name = getattr(league_info, 'name', None)
            if league_name:
                if isinstance(league_name, bytes):
                    league_name = league_name.decode('utf-8')
                self.display_league_name = str(league_name)
                self.context_parts.append(f"  League Name: {league_name}")
        
        if hasattr(league_info, 'league_key'):
            league_key = getattr(league_info, 'league_key', None)
            if league_key:
                self.context_parts.append(f"  League Key: {league_key}")
        
        if hasattr(league_info, 'season'):
            season = getattr(league_info, 'season', None)
            if season:
                self.context_parts.append(f"  Season: {season}")
        
        if hasattr(league_info, 'num_teams'):
            num_teams = getattr(league_info, 'num_teams', None)
            if num_teams:
                self.context_parts.append(f"  Number of Teams: {num_teams}")
    
    def _add_team_standings(self) -> None:
        """Add team standings to context."""
        if not self.matching_team_id:
            self.context_parts.append("Team Standings: Not available (team ID not found)")
            return
        
        try:
            team_standings_result = self.query.get_team_standings(int(self.matching_team_id))
            if team_standings_result:
                self.context_parts.append("Team Standings:")
                standings_data = self._extract_standings_data(team_standings_result)
                
                if standings_data:
                    rank, wins, losses, ties = self._parse_standings_data(standings_data)
                    self.display_rank = str(rank)
                    if wins != 'N/A' or losses != 'N/A' or ties != 'N/A':
                        self.display_record = f"{wins}-{losses}-{ties}"
                    self.context_parts.append(f"  Rank: {rank}")
                    if wins != 'N/A' or losses != 'N/A' or ties != 'N/A':
                        self.context_parts.append(f"  Record: {wins}-{losses}-{ties}")
                else:
                    self.context_parts.append("  Rank: N/A")
            else:
                self.context_parts.append("Team Standings: Not available")
        except Exception as e:
            logger.warning(f"Could not fetch team standings: {e}")
            self.context_parts.append("Team Standings: Error fetching standings")
    
    def _extract_standings_data(self, team_standings_result: Any) -> Optional[Any]:
        """
        Extract standings data from team standings result.
        
        Args:
            team_standings_result: Result from get_team_standings API call
            
        Returns:
            Standings data object/dict, or None if not found
        """
        # Method 1: Check if result has team_standings attribute
        if hasattr(team_standings_result, 'team_standings'):
            return getattr(team_standings_result, 'team_standings', None)
        
        # Method 2: Check if result itself is the standings dict/object
        if hasattr(team_standings_result, 'rank') or (isinstance(team_standings_result, dict) and 'rank' in team_standings_result):
            return team_standings_result
        
        return None
    
    def _parse_standings_data(self, standings_data: Any) -> tuple:
        """
        Parse standings data to extract rank, wins, losses, ties.
        
        Args:
            standings_data: Standings data object or dict
            
        Returns:
            Tuple of (rank, wins, losses, ties)
        """
        rank = 'N/A'
        wins = 'N/A'
        losses = 'N/A'
        ties = 'N/A'
        
        # If it's a dict
        if isinstance(standings_data, dict):
            rank = standings_data.get('rank', 'N/A')
            outcome_totals = standings_data.get('outcome_totals', {})
            if isinstance(outcome_totals, dict):
                wins = outcome_totals.get('wins', 'N/A')
                losses = outcome_totals.get('losses', 'N/A')
                ties = outcome_totals.get('ties', 'N/A')
        # If it's an object with attributes
        else:
            rank = getattr(standings_data, 'rank', None) or (standings_data.get('rank') if hasattr(standings_data, 'get') else 'N/A')
            outcome_totals = getattr(standings_data, 'outcome_totals', None)
            if outcome_totals:
                if isinstance(outcome_totals, dict):
                    wins = outcome_totals.get('wins', 'N/A')
                    losses = outcome_totals.get('losses', 'N/A')
                    ties = outcome_totals.get('ties', 'N/A')
                else:
                    wins = getattr(outcome_totals, 'wins', 'N/A')
                    losses = getattr(outcome_totals, 'losses', 'N/A')
                    ties = getattr(outcome_totals, 'ties', 'N/A')
        
        return (rank, wins, losses, ties)
    
    def _add_current_roster(self) -> None:
        """Add current roster to context using current date."""
        if not self.matching_team_id:
            self.context_parts.append("Current Roster: Not available (team ID not found)")
            return
        
        try:
            players = self._fetch_roster_players()
            
            if players:
                self.context_parts.append("Current Roster:")
                self._format_roster_players(players)
            else:
                self.context_parts.append("Current Roster: Not available")
        except Exception as e:
            logger.warning(f"Could not fetch current team roster: {e}")
            self.context_parts.append(f"Current Roster: Error fetching roster - {str(e)}")
    
    def _fetch_roster_players(self) -> Optional[List[Any]]:
        """
        Fetch roster players using date-based methods, with week-based fallback if needed.
        
        Returns:
            List of player objects, or None if all methods fail
        """
        players = None
        last_error = None
        today_date = datetime.now().strftime("%Y-%m-%d")
        
        # Method 1: Try get_team_roster_player_info_by_date (for NBA/NHL/MLB - preferred method)
        if self.query.game_code in ['nba', 'nhl', 'mlb']:
            try:
                players = self.query.get_team_roster_player_info_by_date(int(self.matching_team_id), today_date)
                logger.debug(f"Successfully fetched roster using get_team_roster_player_info_by_date with date {today_date}")
                if players:
                    return players
            except Exception as e1:
                logger.debug(f"get_team_roster_player_info_by_date failed: {e1}, trying other methods")
                last_error = e1
        
        # Method 2: Fallback to week-based methods only if date method failed and we have current_week
        if not players and self.current_week:
            try:
                week_num = int(self.current_week) if self.current_week else None
                if week_num:
                    # Try get_team_roster_player_info_by_week
                    try:
                        players = self.query.get_team_roster_player_info_by_week(int(self.matching_team_id), week_num)
                        logger.debug(f"Successfully fetched roster using get_team_roster_player_info_by_week")
                        if players:
                            return players
                    except Exception as e2:
                        logger.debug(f"get_team_roster_player_info_by_week failed: {e2}, trying other methods")
                        last_error = e2
                    
                    # Try get_team_roster_player_stats_by_week
                    try:
                        players = self.query.get_team_roster_player_stats_by_week(int(self.matching_team_id), week_num)
                        logger.debug(f"Successfully fetched roster using get_team_roster_player_stats_by_week")
                        if players:
                            return players
                    except Exception as e3:
                        logger.debug(f"get_team_roster_player_stats_by_week failed: {e3}, trying get_team_roster_by_week")
                        last_error = e3
                    
                    # Fallback to get_team_roster_by_week
                    try:
                        roster = self.query.get_team_roster_by_week(int(self.matching_team_id), week_num)
                        if roster:
                            players = self._extract_players_from_roster(roster)
                            logger.debug(f"Successfully fetched roster using get_team_roster_by_week")
                            if players:
                                return players
                    except Exception as e4:
                        logger.warning(f"All roster methods failed. Last error: {e4}")
                        if last_error:
                            logger.warning(f"Previous errors: {last_error}")
                        raise e4
            except Exception as e:
                logger.warning(f"Error in week-based roster fallback: {e}")
        
        if not players and last_error:
            logger.warning(f"All roster methods failed. Last error: {last_error}")
        
        return None
    
    def _extract_players_from_roster(self, roster: Any) -> List[Any]:
        """
        Extract players list from roster object.
        
        Args:
            roster: Roster object from Yahoo API
            
        Returns:
            List of player objects
        """
        players = []
        
        if not hasattr(roster, 'players'):
            return players
        
        players_attr = getattr(roster, 'players', None)
        if not players_attr:
            return players
        
        if isinstance(players_attr, list):
            for item in players_attr:
                if hasattr(item, 'player'):
                    players.append(getattr(item, 'player'))
                else:
                    players.append(item)
        else:
            if hasattr(players_attr, 'player'):
                players = [getattr(players_attr, 'player')]
            else:
                players = [players_attr]
        
        return players
    
    def _format_roster_players(self, players: List[Any]) -> None:
        """
        Format and add roster players to context.
        
        Args:
            players: List of player objects
        """
        # Ensure players is a list
        if not isinstance(players, list):
            players = [players] if players else []
        
        if not players:
            self.context_parts.append("  No players found in roster")
            return
        
        # Format first 15 players; store first 8 for display preview
        for i, player in enumerate(players[:15]):
            try:
                player_entry = self._format_player_entry(player)
                self.context_parts.append(player_entry)
                if i < 8:
                    # Store "Name (Pos)" for display (without "  - " prefix)
                    display_entry = player_entry.replace("  - ", "", 1)
                    self.display_roster_preview.append(display_entry)
            except Exception as e:
                logger.debug(f"Error processing player in roster: {e}")
                self.context_parts.append("  - Unknown Player")
    
    def _format_player_entry(self, player: Any) -> str:
        """
        Format a single player entry for context.
        
        Args:
            player: Player object from Yahoo API
            
        Returns:
            Formatted player entry string
        """
        # Extract player name
        player_name = self._extract_player_name(player)
        
        # Extract position
        position = self._extract_player_position(player)
        
        # Format entry
        return f"  - {player_name} ({position})"
    
    def _extract_player_name(self, player: Any) -> str:
        """
        Extract player name from player object.
        
        Args:
            player: Player object from Yahoo API
            
        Returns:
            Player name as string
        """
        player_name = 'Unknown'
        
        if hasattr(player, 'name'):
            name_obj = getattr(player, 'name', None)
            if name_obj:
                if isinstance(name_obj, dict):
                    player_name = name_obj.get('full', name_obj.get('first', 'Unknown'))
                elif hasattr(name_obj, 'full'):
                    player_name = str(getattr(name_obj, 'full', 'Unknown'))
                elif hasattr(name_obj, 'get'):
                    player_name = str(name_obj.get('full', 'Unknown'))
            else:
                player_name = str(getattr(player, 'name', 'Unknown'))
        
        return player_name
    
    def _extract_player_position(self, player: Any) -> str:
        """
        Extract player position from player object.
        
        Args:
            player: Player object from Yahoo API
            
        Returns:
            Player position as string
        """
        position = getattr(player, 'display_position', None) or getattr(player, 'position', None) or 'N/A'
        if isinstance(position, bytes):
            position = position.decode('utf-8')
        return position
    
    def _add_team_not_found_message(self) -> None:
        """Add message to context when team is not found."""
        self.context_parts.append("=== USER TEAM INFO ===")
        self.context_parts.append(f"Team Name: {self.team_name}")
        self.context_parts.append("Status: Team not found in league")
        self.context_parts.append("")
        logger.debug(f"Team '{self.team_name}' not found in league teams")


def gather_user_context(
    query: YahooFantasySportsQuery,
    team_name: Optional[str] = None
) -> str:
    """
    Convenience function to gather user context.
    
    This function maintains backward compatibility with existing code.
    It creates a UserContextGatherer instance and calls gather().
    
    Args:
        query: YahooFantasySportsQuery instance for API access
        team_name: Name of the user's team (optional)
    
    Returns:
        Formatted context string with current date, user teams, leagues, rosters, and standings
    """
    gatherer = UserContextGatherer(query, team_name)
    return gatherer.gather()
