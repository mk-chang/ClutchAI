"""
ClutchAI Agent - A ReACT AI Agent for Yahoo Fantasy Basketball League Management

This agent combines:
1. Yahoo Fantasy Sports API integration for live league data
2. Local ChromaDB vectorstore with Locked On Basketball Podcast transcripts for draft advice before using this agent.
"""

import os
import yaml
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool, StructuredTool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from yfpy.query import YahooFantasySportsQuery

from ClutchAI.logger import get_logger, setup_logging
from ClutchAI.rag.vector_manager import VectorstoreManager
from ClutchAI.tools.yahoo_api import YahooFantasyTool
from ClutchAI.tools.nba_api import nbaAPITool
from ClutchAI.tools.fantasy_news import FantasyNewsTool
from ClutchAI.tools.dynasty_ranking import DynastyRankingTool

logger = get_logger(__name__)

class ClutchAIAgent:
    """
    Main agent class that combines Yahoo Fantasy API and podcast knowledge retrieval.
    """
    
    def __init__(
        self,
        yahoo_league_id: int = 58930,
        yahoo_client_id: Optional[str] = None,
        yahoo_client_secret: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        env_file_location: Optional[Path] = None,
        chroma_persist_directory: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0,
        debug: bool = False,
    ):
        """
        Initialize the ClutchAI Agent.
        
        Args:
            yahoo_league_id: Yahoo Fantasy League ID
            yahoo_client_id: Yahoo OAuth Client ID (or from env)
            yahoo_client_secret: Yahoo OAuth Client Secret (or from env)
            openai_api_key: OpenAI API key (or from env)
            env_file_location: Path to .env file location
            chroma_persist_directory: Directory to persist ChromaDB (defaults to ClutchAI/rag/chroma_db)
            model_name: OpenAI model to use
            temperature: Temperature for LLM
            debug: Enable debug mode for verbose logging
        """
        # Store debug mode and setup logging
        self.debug = debug
        setup_logging(debug=debug)
        # Set environment file location
        if env_file_location is None:
            # Default to project root (parent of agent/)
            self.env_file_location = Path(__file__).parent.parent.resolve()
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
        
        # Set default ChromaDB persist directory if not provided (store in rag directory)
        if chroma_persist_directory is None:
            self.chroma_persist_directory = str(Path(__file__).parent / "rag" / "chroma_db")
        else:
            self.chroma_persist_directory = chroma_persist_directory
        
        # Initialize and update vectorstore from local persistence
        self.vectorstore = self._initialize_vectorstore()
        self.retriever = self.vectorstore.as_retriever()
        
        # Load agent configuration
        self.agent_config = self._load_agent_config()
        
        # Note: LangSmith tracing is automatically enabled if LANGCHAIN_TRACING_V2=true 
        # and LANGSMITH_API_KEY are set in environment variables.
        # View traces at https://smith.langchain.com/ - you'll see all tool calls, inputs, and responses there.
        
        # Create tools
        self.tools = self._create_tools()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=self.openai_api_key,
        )
        
        # Get system prompt from config or use default
        system_prompt = self.agent_config.get('system_prompt', 'You are a helpful assistant for a Yahoo Fantasy Sports league manager.')
        
        # Create agent
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt,
        )
    
    def _initialize_vectorstore(self) -> Chroma:
        """
        Initialize ChromaDB vectorstore from local persistence and display available content.
        
        This method will:
        1. Check if vectorstore exists
        2. Display statistics about available content
        3. Return the vectorstore instance (or create empty one if needed)
        
        Returns:
            Chroma vectorstore instance
            
        Raises:
            RuntimeError: If vectorstore initialization fails
        """
        # Get path to vector_data.yaml (should be in rag directory)
        vectordata_yaml = Path(__file__).parent / "rag" / "vector_data.yaml"
        
        # Initialize VectorstoreManager to check vectorstore
        vectorstore_manager = VectorstoreManager(
            vectordata_yaml=str(vectordata_yaml),
            chroma_persist_directory=self.chroma_persist_directory,
            openai_api_key=self.openai_api_key,
            env_file_location=self.env_file_location,
        )
        
        # Get vectorstore statistics
        stats = vectorstore_manager.get_vectorstore_stats()
        
        # Get the vectorstore instance (needed for both debug output and return)
        vectorstore = vectorstore_manager.get_vectorstore()
        
        # Display vectorstore information (logger handles debug level filtering)
        logger.debug("\n" + "="*60)
        logger.debug("Vectorstore Status")
        logger.debug("="*60)
        
        if stats.get('exists', False):
            logger.debug(f"✓ Vectorstore exists at: {self.chroma_persist_directory}")
            logger.debug(f"  Total documents: {stats.get('document_count', 0):,}")
            logger.debug(f"  YouTube videos: {stats.get('youtube_urls', 0)}")
            logger.debug(f"  Articles: {stats.get('article_urls', 0)}")
            logger.debug(f"  Total resources: {stats.get('urls_in_vectorstore', 0)}")
            
            # Get detailed information about resources (only if debug mode to avoid expensive query)
            if self.debug and vectorstore is not None:
                try:
                    results = vectorstore._collection.get()
                    if results and 'metadatas' in results and results['metadatas']:
                        # Group by resource_id to get unique resources
                        resources = {}
                        for metadata in results['metadatas']:
                            if metadata:
                                resource_id = metadata.get('resource_id')
                                source_type = metadata.get('source_type', 'unknown')
                                title = metadata.get('title', 'Untitled')
                                url = metadata.get('url', '')
                                
                                if resource_id:
                                    if resource_id not in resources:
                                        resources[resource_id] = {
                                            'source_type': source_type,
                                            'title': title,
                                            'url': url,
                                            'chunks': 0
                                        }
                                    resources[resource_id]['chunks'] += 1
                        
                        if resources:
                            logger.debug(f"\n  Resources in vectorstore ({len(resources)} unique):")
                            youtube_count = 0
                            article_count = 0
                            for resource_id, info in sorted(resources.items(), key=lambda x: x[1]['source_type']):
                                source_type = info['source_type']
                                if source_type == 'youtube':
                                    youtube_count += 1
                                    logger.debug(f"    [{youtube_count}] {info['title']}")
                                    logger.debug(f"        Type: YouTube | Chunks: {info['chunks']} | ID: {resource_id[:12]}...")
                                elif source_type == 'article':
                                    article_count += 1
                                    logger.debug(f"    [{article_count}] {info['title']}")
                                    logger.debug(f"        Type: Article | Chunks: {info['chunks']} | ID: {resource_id[:12]}...")
                except Exception as e:
                    logger.warning(f"Could not get detailed resource information: {e}")
        else:
            logger.debug(f"✗ Vectorstore does not exist at: {self.chroma_persist_directory}")
            logger.debug("  Vectorstore will be created when first document is added.")
            if stats.get('error'):
                logger.error(f"  Error: {stats['error']}")
        
        logger.debug("="*60 + "\n")
        
        if vectorstore is None:
            # Try to create an empty vectorstore as fallback
            try:
                vectorstore = Chroma(
                    persist_directory=self.chroma_persist_directory,
                    embedding_function=OpenAIEmbeddings(api_key=self.openai_api_key),
                )
                if self.debug:
                    logger.debug("Created empty vectorstore (will be populated when documents are added).")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize ChromaDB vectorstore at {self.chroma_persist_directory}: {e}"
                ) from e
        
        return vectorstore
    
    def _load_agent_config(self) -> dict:
        """
        Load agent configuration from agent_config.yaml.
        
        Returns:
            Dictionary with agent configuration
        """
        config_path = Path(__file__).parent / "agent_config.yaml"
        
        if not config_path.exists():
            # Return default config if file doesn't exist
            return {
                'system_prompt': 'You are a helpful assistant for a Yahoo Fantasy Sports league manager.',
                'yahoo_fantasy_news_urls': []
            }
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            
            # Ensure required keys exist with defaults
            if 'system_prompt' not in config:
                config['system_prompt'] = 'You are a helpful assistant for a Yahoo Fantasy Sports league manager.'
            if 'yahoo_fantasy_news_urls' not in config:
                config['yahoo_fantasy_news_urls'] = []
            
            return config
        except Exception as e:
            logger.warning(f"Error loading agent_config.yaml: {e}. Using defaults.")
            return {
                'system_prompt': 'You are a helpful assistant for a Yahoo Fantasy Sports league manager.',
                'yahoo_fantasy_news_urls': []
            }
    
    def _create_tools(self):
        """
        Create LangChain tools for the agent.
        
        Includes all Yahoo Fantasy Sports API tools:
        - Game tools (10): game keys, game info, metadata, weeks, stat categories, positions
        - User tools (4): current user, user games, leagues, teams
        - League tools (12): league key, info, metadata, settings, standings, teams, players, 
          draft results, transactions, scoreboard, matchups
        - Team tools (13): team info, metadata, stats, standings, roster, player info, 
          draft results, matchups
        - Player tools (6): player stats (season/week/date), ownership, percent owned, draft analysis
        
        Includes NBA API tools:
        - Static Data tools (4): get all players/teams, find players/teams by name
        - Player Stats tools (5): player info, career stats, game log, dashboard splits
        - Team Stats tools (2): team info, game log
        - Game tools (5): scoreboard, live scoreboard, box score, play-by-play, find games
        
        Includes Yahoo Fantasy News tools:
        - Web scraping tools (3-4): scrape Yahoo Fantasy NBA news, scrape any URL, map URL, 
          and optionally map all configured URLs
        
        Includes Dynasty Rankings tools:
        - Dynasty ranking tools (3): get player dynasty rank, search players by name, refresh rankings
        
        - Vectorstore tool (1): retrieve contextual knowledge from vectorstore
        
        Returns:
            List of tool instances (45+ Yahoo Fantasy tools + 16 NBA API tools + 4-5 Yahoo Fantasy News tools + 3 Dynasty Ranking tools + 1 vectorstore tool)
        """
        # Create Yahoo Fantasy tools using YahooFantasyTool class
        # This includes all 45 Yahoo Fantasy Sports API tools
        yahoo_tool = YahooFantasyTool(self.query, debug=self.debug)
        yahoo_tools = yahoo_tool.get_all_tools()
        
        # Create NBA API tools using nbaAPITool class
        # This includes all 16 NBA API tools
        nba_tool = nbaAPITool(debug=self.debug)
        nba_tools = nba_tool.get_all_tools()
        
        # Create Yahoo Fantasy News tools
        news_urls = self.agent_config.get('yahoo_fantasy_news_urls', [])
        yahoo_news_tool = FantasyNewsTool(urls=news_urls, debug=self.debug)
        yahoo_news_tools = yahoo_news_tool.get_all_tools()
        
        # Create Dynasty Ranking tools
        dynasty_ranking_tool = DynastyRankingTool(debug=self.debug)
        dynasty_ranking_tools = dynasty_ranking_tool.get_all_tools()
        
        def retrieve_vectorstore_func(query: str) -> str:
            """Retrieve contextual knowledge from the vectorstore."""
            try:
                results = self.retriever.invoke(query)
                return "\n\n".join([r.page_content for r in results])
            except Exception as e:
                return f"Failed to retrieve knowledge: {e}"
        
        retrieve_vectorstore = StructuredTool.from_function(
            func=retrieve_vectorstore_func,
            name="vectorstore_retriever",
            description="Retrieve contextual knowledge from the vectorstore."
        )
        
        # Ensure all tool lists are lists (not None)
        yahoo_tools = yahoo_tools or []
        nba_tools = nba_tools or []
        yahoo_news_tools = yahoo_news_tools or []
        dynasty_ranking_tools = dynasty_ranking_tools or []
        
        tools = yahoo_tools + nba_tools + yahoo_news_tools + dynasty_ranking_tools + [retrieve_vectorstore]
        
        # Log all tools
        self._log_all_tools(tools)
        
        return tools

    def _log_all_tools(self, tools):
        """
        Log all available tool names.
        
        Args:
            tools: List of all tools
        """
        logger.info("\n" + "="*60)
        logger.info(f"Available Tools ({len(tools)}):")
        logger.info("="*60)
        
        for i, tool in enumerate(tools, 1):
            try:
                tool_name = getattr(tool, 'name', getattr(tool, '__name__', f'<unnamed tool {i}>'))
                logger.info(f"  [{i:3d}] {tool_name}")
            except Exception as e:
                logger.error(f"  [{i:3d}] Error getting tool name: {e}")
        
        logger.info("="*60 + "\n")

    def invoke(self, messages: list) -> dict:
        """
        Invoke the agent with a list of messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys, 
                     or LangChain message objects (HumanMessage, AIMessage, etc.)
            
        Returns:
            Agent response dictionary
        """
        # Convert dict format to LangChain messages if needed
        from langchain_core.messages import HumanMessage, AIMessage
        langchain_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                if msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_messages.append(AIMessage(content=msg["content"]))
            else:
                # Already a LangChain message object
                langchain_messages.append(msg)
        
        inputs = {"messages": langchain_messages}
        return self.agent.invoke(inputs)
    
    def stream(self, messages: list):
        """
        Stream the agent response.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys,
                     or LangChain message objects (HumanMessage, AIMessage, etc.)
            
        Yields:
            Agent response events
        """
        # Convert dict format to LangChain messages if needed
        from langchain_core.messages import HumanMessage, AIMessage
        langchain_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                if msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_messages.append(AIMessage(content=msg["content"]))
            else:
                # Already a LangChain message object
                langchain_messages.append(msg)
        
        inputs = {"messages": langchain_messages}
        for event in self.agent.stream(inputs):
            yield event
    
    def chat(self, user_message: str, conversation_history: Optional[list] = None) -> str:
        """
        Simple chat interface that returns the final response as a string.
        
        Args:
            user_message: User's message string
            conversation_history: Optional list of previous messages with 'role' and 'content' keys
            
        Returns:
            Agent's response string
        """
        # Build messages list
        if conversation_history:
            messages = conversation_history + [{"role": "user", "content": user_message}]
        else:
            messages = [{"role": "user", "content": user_message}]
        
        response = self.invoke(messages)
        
        # Extract the final AI message from the response
        for msg in reversed(response.get("messages", [])):
            if hasattr(msg, 'content') and msg.content and msg.content.strip():
                return msg.content
        
        return "I apologize, but I couldn't generate a response. Please try again."

