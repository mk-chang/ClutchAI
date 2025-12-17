"""
Multi-Agent System for ClutchAI

This module provides a multi-agent system with:
- Supervisor Agent: Orchestrates workflow
- Research Agent: Gathers data from all sources
- Analysis Agent: Analyzes data and generates recommendations
"""

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

from yfpy.query import YahooFantasySportsQuery

from logger import get_logger, setup_logging
from agents.multi_agent.supervisor import SupervisorAgent
from agents.multi_agent.research_agent import ResearchAgent
from agents.multi_agent.analysis_agent import AnalysisAgent
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
            model_name: Not used (models configured in multi_agent_config.yaml)
            temperature: Not used (temperatures configured in multi_agent_config.yaml)
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
        
        # Load tools configuration
        self.tools_config = self._load_tools_config()
        
        # Create Research Agent
        self.research_agent = ResearchAgent(
            query=self.query,
            rag_manager=self.rag_manager,
            tools_config=self.tools_config,
            openai_api_key=self.openai_api_key,
            project_root=self.env_file_location,
            debug=self.debug,
        )
        
        # Create Analysis Agent
        self.analysis_agent = AnalysisAgent(
            openai_api_key=self.openai_api_key,
            project_root=self.env_file_location,
            debug=self.debug,
        )
        
        # Create Supervisor Agent
        self.supervisor = SupervisorAgent(
            research_agent=self.research_agent,
            analysis_agent=self.analysis_agent,
            openai_api_key=self.openai_api_key,
            project_root=self.env_file_location,
            debug=self.debug,
        )
        
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
    
    def chat(self, query: str, conversation_history: Optional[list] = None) -> str:
        """
        Chat with the multi-agent system, supporting conversation history.
        
        Args:
            query: User query/question
            conversation_history: List of message dicts with 'role' and 'content' keys
                                 (e.g., [{"role": "user", "content": "..."}, ...])
            
        Returns:
            Agent response as a string
        """
        # Build messages list from conversation history if provided
        messages = []
        if conversation_history:
            for msg in conversation_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in ("user", "assistant"):
                    messages.append((role, content))
        
        # Add the current query
        messages.append(("user", query))
        
        # Invoke supervisor with full conversation history
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

