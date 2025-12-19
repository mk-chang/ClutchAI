"""
Multi-Agent System for ClutchAI

This module provides a multi-agent system with:
- Supervisor Agent: Orchestrates workflow
- Yahoo Fantasy Agent: Gathers Yahoo Fantasy API data
- Statistic Agent: Gathers NBA statistics and game data
- News Agent: Gathers news, insights, and contextual information
- Analysis Agent: Analyzes data and generates recommendations
"""

import os
import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import tiktoken

from yfpy.query import YahooFantasySportsQuery

from logger import get_logger, setup_logging
from agents.multi_agent.base_agent import BaseAgent
from agents.multi_agent.supervisor import SupervisorAgent
from agents.multi_agent.yahoo_fantasy_agent import YahooFantasyAgent
from agents.multi_agent.statistic_agent import StatisticAgent
from agents.multi_agent.news_agent import NewsAgent
from agents.multi_agent.analyst_agent import FantasyAnalystAgent
from agents.rag.rag_manager import RAGManager
from data.cloud_sql.connection import PostgresConnection
from data.cloud_sql.schema import get_default_table_name

logger = get_logger(__name__)


class MultiAgentSystem:
    """
    Main entry point for the multi-agent system.
    
    Provides the same interface as ClutchAIAgent for compatibility.
    """
    
    def __init__(
        self,
        yahoo_league_id: int = 58930,
        yahoo_client_id: Optional[str] = None,
        yahoo_client_secret: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        env_file_location: Optional[Path] = None,
        connection: Optional[PostgresConnection] = None,
        table_name: Optional[str] = None,
        team_name: Optional[str] = None,
        model_name: str = "gpt-4o-mini",  # Not used, but kept for compatibility
        temperature: float = 0,  # Not used, but kept for compatibility
        debug: bool = False,
    ):
        """
        Initialize the Multi-Agent System.
        
        Args:
            yahoo_league_id: Yahoo Fantasy League ID
            yahoo_client_id: Yahoo OAuth Client ID (or from env)
            yahoo_client_secret: Yahoo OAuth Client Secret (or from env)
            openai_api_key: OpenAI API key (or from env)
            env_file_location: Path to .env file location
            connection: PostgresConnection instance (optional)
            table_name: Name of the vector table in PostgreSQL
            team_name: User's fantasy team name (optional, defaults to "KATmandu Climbers")
            model_name: Not used (models configured in multiagent_config.yaml)
            temperature: Not used (temperatures configured in multiagent_config.yaml)
            debug: Enable debug mode for verbose logging
        """
        # Store debug mode and setup logging
        self.debug = debug
        setup_logging(debug=debug)
        
        # Set environment file location
        if env_file_location is None:
            self.env_file_location = Path(__file__).parent.parent.parent.resolve()
        else:
            self.env_file_location = Path(env_file_location)
        
        # Set API keys from parameters or environment
        self.yahoo_client_id = yahoo_client_id or os.environ.get('YAHOO_CLIENT_ID')
        self.yahoo_client_secret = yahoo_client_secret or os.environ.get('YAHOO_CLIENT_SECRET')
        self.openai_api_key = openai_api_key or os.environ.get('OPENAI_API_KEY')
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY env var or pass openai_api_key parameter.")
        
        # Set Yahoo Fantasy parameters
        self.yahoo_league_id = yahoo_league_id
        self.game_code = "nba"
        self.game_id = 466
        
        # Initialize Yahoo Fantasy query
        self.query = YahooFantasySportsQuery(
            league_id=self.yahoo_league_id,
            game_code=self.game_code,
            game_id=self.game_id,
            yahoo_consumer_key=self.yahoo_client_id,
            yahoo_consumer_secret=self.yahoo_client_secret,
            env_var_fallback=True,
            env_file_location=self.env_file_location,
            save_token_data_to_env_file=True,
        )
        
        # Initialize RAG manager
        self.table_name = table_name or get_default_table_name()
        self.rag_manager = RAGManager(
            connection=connection,
            table_name=self.table_name,
            openai_api_key=self.openai_api_key,
            project_root=self.env_file_location,
        )
        
        # Store team name
        self.team_name = team_name or "KATmandu Climbers"
        
        # Load tools configuration
        self.tools_config = self._load_tools_config()
        
        # Gather user context (teams, leagues, rosters) for supervisor
        user_context = self._gather_user_context()
        
        # Log user context in debug mode
        if user_context:
            logger.debug(f"User context gathered:\n{user_context}")
        
        # Create specialized Research Agents
        self.yahoo_fantasy_agent = YahooFantasyAgent(
            query=self.query,
            rag_manager=None,  # Yahoo agent doesn't need RAG
            tools_config=self.tools_config,
            openai_api_key=self.openai_api_key,
            project_root=self.env_file_location,
            debug=self.debug,
        )
        
        self.statistic_agent = StatisticAgent(
            query=None,  # Statistic agent doesn't need Yahoo query
            rag_manager=None,  # Statistic agent doesn't need RAG
            tools_config=self.tools_config,
            openai_api_key=self.openai_api_key,
            project_root=self.env_file_location,
            debug=self.debug,
        )
        
        self.news_agent = NewsAgent(
            query=None,  # News agent doesn't need Yahoo query
            rag_manager=self.rag_manager,
            tools_config=self.tools_config,
            openai_api_key=self.openai_api_key,
            project_root=self.env_file_location,
            debug=self.debug,
        )
        
        # Create Analysis Agent
        self.analysis_agent = FantasyAnalystAgent(
            rag_manager=self.rag_manager,
            user_context=user_context,
            openai_api_key=self.openai_api_key,
            project_root=self.env_file_location,
            debug=self.debug,
        )
        
        # Create Supervisor Agent
        self.supervisor = SupervisorAgent(
            yahoo_fantasy_agent=self.yahoo_fantasy_agent,
            statistic_agent=self.statistic_agent,
            news_agent=self.news_agent,
            analysis_agent=self.analysis_agent,
            user_context=user_context,
            openai_api_key=self.openai_api_key,
            project_root=self.env_file_location,
            debug=self.debug,
        )
        
        # Load max_tokens from supervisor config (supervisor is the entry point for chat)
        supervisor_config = self.supervisor.config.get('supervisor', {})
        self.max_tokens = supervisor_config.get('max_tokens', 150000)
        
        logger.info("Multi-Agent System initialized successfully")
    
    def _load_tools_config(self) -> dict:
        """Load tools configuration from tools_config.yaml."""
        config_path = self.env_file_location / "config" / "tools_config.yaml"
        
        if not config_path.exists():
            logger.warning(f"tools_config.yaml not found at {config_path}. Using defaults.")
            return {
                'yahoo_fantasy_news_urls': [],
                'dynasty_rankings_url': [],
                'rotowire_rss_url': None
            }
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            
            # Ensure required keys exist with defaults
            if 'yahoo_fantasy_news_urls' not in config:
                config['yahoo_fantasy_news_urls'] = []
            if 'dynasty_rankings_url' not in config:
                config['dynasty_rankings_url'] = []
            if 'rotowire_rss_url' not in config:
                config['rotowire_rss_url'] = None
            
            return config
        except Exception as e:
            logger.warning(f"Error loading tools_config.yaml: {e}. Using defaults.")
            return {
                'yahoo_fantasy_news_urls': [],
                'dynasty_rankings_url': [],
                'rotowire_rss_url': None
            }
    
    def _gather_user_context(self) -> str:
        """
        Gather context about the user's teams, leagues, and rosters for the current basketball season.
        
        Returns:
            Formatted context string with current date, user teams, leagues, rosters, and standings
        """
        try:
            context_parts = []
            
            # Get current date
            current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            context_parts.append(f"=== CURRENT DATE/TIME ===")
            context_parts.append(f"Current Date/Time: {current_date}")
            context_parts.append("")
            
            # Get basic info about the user's team if team name is provided
            if self.team_name:
                try:
                    # Get league teams to find the matching team
                    league_teams = self.query.get_league_teams()
                    
                    if league_teams:
                        # Find team matching the team name
                        matching_team = None
                        matching_team_id = None
                        matching_team_key = None
                        
                        # Process teams list
                        teams_list = []
                        if isinstance(league_teams, list):
                            teams_list = league_teams
                        elif hasattr(league_teams, 'team'):
                            teams_list = league_teams.team if isinstance(league_teams.team, list) else [league_teams.team]
                        
                        # Search for team by name (case-insensitive)
                        for idx, team in enumerate(teams_list):
                            team_name_attr = None
                            # Try multiple ways to access the name attribute
                            # yfpy Team objects support attribute access via __getattr__, so hasattr may not work
                            team_name_attr = None
                            try:
                                # First try direct attribute access (most common for yfpy)
                                team_name_attr = team.name
                            except (AttributeError, KeyError, TypeError):
                                try:
                                    # Try getattr as fallback
                                    team_name_attr = getattr(team, 'name', None)
                                except (AttributeError, KeyError, TypeError):
                                    pass
                            
                            # Try team_name attribute if name didn't work
                            if not team_name_attr:
                                try:
                                    team_name_attr = getattr(team, 'team_name', None)
                                except (AttributeError, KeyError, TypeError):
                                    pass
                            
                            # Try dictionary-style access (yfpy objects sometimes support this)
                            if not team_name_attr:
                                try:
                                    if hasattr(team, '__getitem__'):
                                        team_name_attr = team['name']
                                except (TypeError, KeyError, AttributeError):
                                    pass
                            
                            # Try accessing via __dict__ or similar
                            if not team_name_attr:
                                try:
                                    if hasattr(team, '__dict__') and 'name' in team.__dict__:
                                        team_name_attr = team.__dict__['name']
                                except (AttributeError, KeyError):
                                    pass
                            
                            # Convert bytes to string if needed (yfpy sometimes returns bytes)
                            if team_name_attr and isinstance(team_name_attr, bytes):
                                try:
                                    team_name_attr = team_name_attr.decode('utf-8')
                                except (UnicodeDecodeError, AttributeError):
                                    # If decode fails, try to convert to string
                                    team_name_attr = str(team_name_attr)
                            
                            # Ensure it's a string for comparison
                            if team_name_attr:
                                team_name_attr = str(team_name_attr)
                            
                            if team_name_attr and team_name_attr.lower().strip() == str(self.team_name).lower().strip():
                                matching_team = team
                                if hasattr(team, 'team_id'):
                                    matching_team_id = getattr(team, 'team_id', None)
                                elif hasattr(team, 'id'):
                                    matching_team_id = getattr(team, 'id', None)
                                
                                if hasattr(team, 'team_key'):
                                    matching_team_key = getattr(team, 'team_key', None)
                                elif hasattr(team, 'key'):
                                    matching_team_key = getattr(team, 'key', None)
                                break
                        
                        # If team found, get basic info
                        if matching_team and matching_team_id:
                            context_parts.append(f"=== USER TEAM INFO ===")
                            context_parts.append(f"Team Name: {self.team_name}")
                            if matching_team_id:
                                context_parts.append(f"Team ID: {matching_team_id}")
                            if matching_team_key:
                                context_parts.append(f"Team Key: {matching_team_key}")
                            context_parts.append("")
                            
                            # Add league information
                            context_parts.append(f"=== LEAGUE INFO ===")
                            
                            # League size (number of teams)
                            league_size = len(teams_list) if teams_list else 0
                            context_parts.append(f"League Size: {league_size} teams")
                            
                            # League scoring type (from team object)
                            try:
                                league_scoring_type = getattr(matching_team, 'league_scoring_type', None)
                                if league_scoring_type:
                                    # Convert bytes to string if needed
                                    if isinstance(league_scoring_type, bytes):
                                        league_scoring_type = league_scoring_type.decode('utf-8')
                                    context_parts.append(f"League Scoring Type: {league_scoring_type}")
                            except Exception:
                                pass
                            
                            # Current week (fetch once and reuse)
                            current_week = None
                            try:
                                game_info = self.query.get_current_game_info()
                                if game_info:
                                    current_week = getattr(game_info, 'current_week', None) or getattr(game_info, 'week', None)
                                if current_week:
                                    context_parts.append(f"Current Week: {current_week}")
                            except Exception as e:
                                logger.debug(f"Could not fetch current week: {e}")
                            
                            # League metadata
                            try:
                                league_info = self.query.get_league_info()
                                if league_info:
                                    context_parts.append("League Metadata:")
                                    # Extract common league info fields
                                    if hasattr(league_info, 'name'):
                                        league_name = getattr(league_info, 'name', None)
                                        if league_name:
                                            if isinstance(league_name, bytes):
                                                league_name = league_name.decode('utf-8')
                                            context_parts.append(f"  League Name: {league_name}")
                                    if hasattr(league_info, 'league_key'):
                                        league_key = getattr(league_info, 'league_key', None)
                                        if league_key:
                                            context_parts.append(f"  League Key: {league_key}")
                                    if hasattr(league_info, 'season'):
                                        season = getattr(league_info, 'season', None)
                                        if season:
                                            context_parts.append(f"  Season: {season}")
                                    if hasattr(league_info, 'num_teams'):
                                        num_teams = getattr(league_info, 'num_teams', None)
                                        if num_teams:
                                            context_parts.append(f"  Number of Teams: {num_teams}")
                            except Exception as e:
                                logger.debug(f"Could not fetch league info: {e}")
                            
                            # Try league metadata as well
                            try:
                                league_metadata = self.query.get_league_metadata()
                                if league_metadata:
                                    # Add any additional metadata fields not already captured
                                    if hasattr(league_metadata, 'name') and not any('League Name' in part for part in context_parts):
                                        league_name = getattr(league_metadata, 'name', None)
                                        if league_name:
                                            if isinstance(league_name, bytes):
                                                league_name = league_name.decode('utf-8')
                                            context_parts.append(f"  League Name: {league_name}")
                            except Exception as e:
                                logger.debug(f"Could not fetch league metadata: {e}")
                            
                            context_parts.append("")
                            
                            # Get team standings
                            try:
                                team_standings = self.query.get_team_standings(int(matching_team_id))
                                if team_standings:
                                    context_parts.append("Team Standings:")
                                    if hasattr(team_standings, 'team_standings'):
                                        standings = getattr(team_standings, 'team_standings', {})
                                        if isinstance(standings, dict):
                                            rank = standings.get('rank', 'N/A')
                                            wins = standings.get('outcome_totals', {}).get('wins', 'N/A') if isinstance(standings.get('outcome_totals'), dict) else 'N/A'
                                            losses = standings.get('outcome_totals', {}).get('losses', 'N/A') if isinstance(standings.get('outcome_totals'), dict) else 'N/A'
                                            ties = standings.get('outcome_totals', {}).get('ties', 'N/A') if isinstance(standings.get('outcome_totals'), dict) else 'N/A'
                                            context_parts.append(f"  Rank: {rank}")
                                            context_parts.append(f"  Record: {wins}-{losses}-{ties}")
                                    elif hasattr(team_standings, 'rank'):
                                        context_parts.append(f"  Rank: {getattr(team_standings, 'rank', 'N/A')}")
                            except Exception as e:
                                logger.debug(f"Could not fetch team standings: {e}")
                            
                            # Get current roster
                            try:
                                # Use current_week already fetched above
                                if current_week:
                                    roster = self.query.get_team_roster_by_week(int(matching_team_id), int(current_week))
                                    if roster:
                                        context_parts.append(f"Current Roster (Week {current_week}):")
                                        if hasattr(roster, 'player'):
                                            players = roster.player if isinstance(roster.player, list) else [roster.player]
                                            for player in players[:15]:  # First 15 players
                                                player_name = getattr(player, 'name', {}).get('full', 'Unknown') if hasattr(getattr(player, 'name', None), 'get') else getattr(player, 'name', 'Unknown')
                                                position = getattr(player, 'display_position', 'N/A')
                                                context_parts.append(f"  - {player_name} ({position})")
                                else:
                                    # Fallback: try week 1
                                    roster = self.query.get_team_roster_by_week(int(matching_team_id), 1)
                                    if roster:
                                        context_parts.append("Roster (Week 1):")
                                        if hasattr(roster, 'player'):
                                            players = roster.player if isinstance(roster.player, list) else [roster.player]
                                            for player in players[:15]:
                                                player_name = getattr(player, 'name', {}).get('full', 'Unknown') if hasattr(getattr(player, 'name', None), 'get') else getattr(player, 'name', 'Unknown')
                                                position = getattr(player, 'display_position', 'N/A')
                                                context_parts.append(f"  - {player_name} ({position})")
                            except Exception as e:
                                logger.debug(f"Could not fetch team roster: {e}")
                            
                            context_parts.append("")
                        else:
                            logger.debug(f"Team '{self.team_name}' not found in league teams")
                except Exception as e:
                    logger.debug(f"Could not fetch team info for '{self.team_name}': {e}")
            
            context_str = "\n".join(context_parts)
            logger.info(f"Gathered user context ({len(context_str)} characters)")
            return context_str
            
        except Exception as e:
            logger.error(f"Error gathering user context: {e}")
            return f"Error gathering user context: {e}"
    
    def invoke(self, query: str, **kwargs):
        """
        Invoke the multi-agent system with a query.
        
        Args:
            query: User query/question
            **kwargs: Additional arguments to pass to agent
            
        Returns:
            Agent response
        """
        return self.supervisor.invoke(query, **kwargs)
    
    def _estimate_tokens(self, text: str, model: str = "gpt-4o-mini") -> int:
        """
        Estimate the number of tokens in a text string.
        
        Args:
            text: Text to count tokens for
            model: Model name to use for tokenization
            
        Returns:
            Estimated token count
        """
        try:
            # Get encoding for the model
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except KeyError:
            # Fallback to cl100k_base encoding (used by gpt-4, gpt-3.5-turbo)
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
    
    def _truncate_history_by_tokens(
        self, 
        conversation_history: list, 
        max_tokens: int,
        current_query: str,
        safety_margin: int = 10000
    ) -> list:
        """
        Truncate conversation history to stay within token limits.
        
        Args:
            conversation_history: List of message dicts
            max_tokens: Maximum tokens allowed (including current query)
            current_query: The current user query
            safety_margin: Safety margin to leave for response tokens
            
        Returns:
            Truncated conversation history
        """
        # Estimate tokens for current query
        query_tokens = self._estimate_tokens(current_query)
        available_tokens = max_tokens - query_tokens - safety_margin
        
        if available_tokens <= 0:
            logger.warning(f"Current query alone ({query_tokens} tokens) exceeds available budget. Using empty history.")
            return []
        
        # Build messages and count tokens
        truncated_history = []
        total_tokens = 0
        
        # Process messages in reverse (most recent first) to keep recent context
        for msg in reversed(conversation_history):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role not in ("user", "assistant"):
                continue
            
            # Estimate tokens for this message (format: "role: content")
            message_text = f"{role}: {content}"
            message_tokens = self._estimate_tokens(message_text)
            
            # If adding this message would exceed the limit, stop
            if total_tokens + message_tokens > available_tokens:
                logger.debug(
                    f"Truncating history: {len(truncated_history)} messages, "
                    f"{total_tokens} tokens (limit: {available_tokens})"
                )
                break
            
            # Add to beginning (since we're processing in reverse)
            truncated_history.insert(0, msg)
            total_tokens += message_tokens
        
        # If we have space and there's a first message (greeting), try to keep it
        if truncated_history and len(truncated_history) < len(conversation_history):
            first_msg = conversation_history[0]
            if first_msg not in truncated_history:
                first_msg_text = f"{first_msg.get('role', 'user')}: {first_msg.get('content', '')}"
                first_msg_tokens = self._estimate_tokens(first_msg_text)
                
                # If we can fit the first message, add it
                if total_tokens + first_msg_tokens <= available_tokens:
                    truncated_history.insert(0, first_msg)
                    total_tokens += first_msg_tokens
        
        logger.debug(
            f"History truncation: {len(conversation_history)} -> {len(truncated_history)} messages, "
            f"~{total_tokens} tokens (query: {query_tokens} tokens, limit: {max_tokens})"
        )
        
        return truncated_history
    
    def chat(
        self, 
        query: str, 
        conversation_history: Optional[list] = None, 
        max_history_messages: int = 20
    ) -> str:
        """
        Chat with the multi-agent system, supporting conversation history.
        
        Args:
            query: User query/question
            conversation_history: List of message dicts with 'role' and 'content' keys
                                 (e.g., [{"role": "user", "content": "..."}, ...])
            max_history_messages: Maximum number of messages to include in history (default: 20)
                                  Used as a fallback if token counting fails
            
        Returns:
            Agent response as a string
        """
        # Build messages list from conversation history if provided
        messages = []
        if conversation_history:
            # Truncate history based on token count (more accurate than message count)
            # Use max_tokens from config
            try:
                truncated_history = self._truncate_history_by_tokens(
                    conversation_history, 
                    max_tokens=self.max_tokens,
                    current_query=query
                )
            except Exception as e:
                logger.warning(f"Error truncating by tokens, falling back to message count: {e}")
                # Fallback to message count limit
                if len(conversation_history) > max_history_messages:
                    truncated_history = [conversation_history[0]] + conversation_history[-max_history_messages+1:]
                else:
                    truncated_history = conversation_history
            
            for msg in truncated_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in ("user", "assistant"):
                    messages.append((role, content))
        
        # Add the current query
        messages.append(("user", query))
        
        # Invoke supervisor with conversation history
        inputs = {"messages": messages}
        response = self.supervisor.agent.invoke(inputs)
        
        # Extract text from response
        if isinstance(response, dict):
            # Check for 'output' key first (standard LangChain agent response)
            if "output" in response:
                return str(response["output"])
            # Check for 'messages' key
            elif "messages" in response:
                # Get the last message (assistant's response)
                last_message = response["messages"][-1]
                if hasattr(last_message, "content"):
                    return str(last_message.content)
                elif isinstance(last_message, tuple):
                    return str(last_message[1])  # (role, content) tuple
                else:
                    return str(last_message)
            # Try to get content directly
            elif "content" in response:
                return str(response["content"])
            else:
                return str(response)
        elif isinstance(response, list) and len(response) > 0:
            # If response is a list of messages, get the last one
            last_message = response[-1]
            if hasattr(last_message, "content"):
                return str(last_message.content)
            elif isinstance(last_message, tuple):
                return str(last_message[1])
            else:
                return str(last_message)
        else:
            return str(response)
    
    def stream(self, query: str, **kwargs):
        """
        Stream multi-agent system responses.
        
        Args:
            query: User query/question
            **kwargs: Additional arguments to pass to agent
            
        Yields:
            Agent response events
        """
        inputs = {"messages": [("user", query)], **kwargs}
        for event in self.supervisor.agent.stream(inputs):
            yield event

