"""
ClutchAI Agent - A ReACT AI Agent for Yahoo Fantasy Basketball League Management

This agent combines:
1. Yahoo Fantasy Sports API integration for live league data
2. PostgreSQL/pgvector vectorstore with Locked On Basketball Podcast transcripts for draft advice.
"""

import os
import yaml
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool, StructuredTool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from yfpy.query import YahooFantasySportsQuery

from agents.logger import get_logger, setup_logging
from agents.rag.rag_manager import RAGManager
from agents.tools.yahoo_api import YahooFantasyTool
from agents.tools.nba_api import nbaAPITool
from agents.tools.fantasy_news import FantasyNewsTool
from agents.tools.dynasty_ranking import DynastyRankingTool
from data.cloud_sql.connection import PostgresConnection

logger = get_logger(__name__)

class ClutchAIAgent:
    """
    Main agent class that combines Yahoo Fantasy API and podcast knowledge retrieval.
    
    This agent uses the ReACT (Reasoning + Acting) framework to:
    1. Understand user queries about their fantasy league
    2. Retrieve relevant context from vectorstore (podcast transcripts, articles)
    3. Call appropriate tools (Yahoo API, NBA API, etc.) to get live data
    4. Synthesize responses using LLM with both context and live data
    """
    
    def __init__(
        self,
        yahoo_league_id: int = 58930,
        yahoo_client_id: Optional[str] = None,
        yahoo_client_secret: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        env_file_location: Optional[Path] = None,
        connection: Optional[PostgresConnection] = None,
        table_name: str = "embeddings",
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
            connection: PostgresConnection instance (optional, will create from env vars if not provided)
            table_name: Name of the vector table in PostgreSQL (default: "embeddings")
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
        
        # Initialize RAG manager for vectorstore access
        self.rag_manager = RAGManager(
            connection=connection,
            table_name=table_name,
            openai_api_key=self.openai_api_key,
        )
        
        # Initialize vectorstore and retriever
        self.vectorstore = self._initialize_vectorstore()
        self.retriever = self.rag_manager.retriever
        
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
        
        # Create agent using LangChain's create_agent
        system_prompt = self.agent_config.get('system_prompt', 'You are a helpful assistant for a Yahoo Fantasy Sports league manager.')
        
        # Create the agent
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt,
        )
    
    def _initialize_vectorstore(self):
        """
        Initialize and display vectorstore information.
        
        Returns:
            Vectorstore instance
        """
        # Get stats and vectorstore from RAG manager
        stats = self.rag_manager.get_stats()
        vectorstore = self.rag_manager.vectorstore
        
        # Display vectorstore information (logger handles debug level filtering)
        logger.debug("\n" + "="*60)
        logger.debug("Vectorstore Status")
        logger.debug("="*60)
        
        if 'error' not in stats:
            logger.debug(f"✓ Connected to PostgreSQL vectorstore")
            logger.debug(f"  Table: {stats.get('table_name', 'embeddings')}")
            logger.debug(f"  Total documents: {stats.get('row_count', 0):,}")
            logger.debug(f"  Unique resources: {stats.get('unique_resources', 0)}")
            
            # Display source type distribution
            source_types = stats.get('source_types', {})
            if source_types:
                logger.debug(f"  Source types:")
                for source_type, count in source_types.items():
                    logger.debug(f"    - {source_type}: {count} documents")
            
            # Get detailed information about resources (only if debug mode)
            if self.debug:
                try:
                    # Query for unique resources with counts
                    from data.cloud_sql.connection import PostgresConnection
                    connection = PostgresConnection()
                    
                    results = connection.execute(
                        f"""
                        SELECT 
                            resource_id,
                            source_type,
                            title,
                            url,
                            COUNT(*) as chunks
                        FROM {stats.get('table_name', 'embeddings')}
                        WHERE resource_id IS NOT NULL
                        GROUP BY resource_id, source_type, title, url
                        ORDER BY source_type, title
                        LIMIT 50
                        """
                    )
                    
                    if results:
                        logger.debug(f"\n  Resources in vectorstore (showing up to 50):")
                        youtube_count = 0
                        article_count = 0
                        for row in results:
                            source_type = row.get('source_type', 'unknown')
                            title = row.get('title', 'Untitled')
                            resource_id = row.get('resource_id', '')
                            chunks = row.get('chunks', 0)
                            
                            if source_type == 'youtube':
                                youtube_count += 1
                                logger.debug(f"    [{youtube_count}] {title}")
                                logger.debug(f"        Type: YouTube | Chunks: {chunks} | ID: {resource_id[:12]}...")
                            elif source_type == 'article':
                                article_count += 1
                                logger.debug(f"    [{article_count}] {title}")
                                logger.debug(f"        Type: Article | Chunks: {chunks} | ID: {resource_id[:12]}...")
                except Exception as e:
                    logger.warning(f"Could not get detailed resource information: {e}")
        else:
            logger.debug(f"✗ Error connecting to vectorstore: {stats.get('error', 'Unknown error')}")
            if stats.get('error'):
                logger.error(f"  Error: {stats['error']}")
        
        logger.debug("="*60 + "\n")
        
        if vectorstore is None:
            raise RuntimeError(
                f"Failed to initialize PostgreSQL vectorstore: {stats.get('error', 'Unknown error')}"
            )
        
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
        Create and return list of tools for the agent.
        
        Returns:
            List of LangChain tools
        """
        tools = []
        
        # Yahoo Fantasy API tools
        try:
            yahoo_tool = YahooFantasyTool(
                query=self.query,
                yahoo_league_id=self.yahoo_league_id,
            )
            tools.extend(yahoo_tool.get_tools())
        except Exception as e:
            logger.warning(f"Yahoo Fantasy tools not available: {e}")
        
        # NBA API tools
        try:
            nba_tool = nbaAPITool()
            tools.extend(nba_tool.get_tools())
        except Exception as e:
            logger.warning(f"NBA API tools not available: {e}")
        
        # Fantasy News tools
        try:
            news_urls = self.agent_config.get('yahoo_fantasy_news_urls', [])
            if news_urls:
                fantasy_news_tool = FantasyNewsTool(urls=news_urls)
                tools.extend(fantasy_news_tool.get_tools())
        except Exception as e:
            logger.warning(f"Fantasy News tools not available: {e}")
        
        # Dynasty Ranking tools
        try:
            dynasty_tool = DynastyRankingTool()
            tools.extend(dynasty_tool.get_tools())
        except Exception as e:
            logger.warning(f"Dynasty Ranking tools not available: {e}")
        
        # RAG retrieval tool (for podcast/article context)
        try:
            @tool
            def search_knowledge_base(query: str) -> str:
                """
                Search the knowledge base (podcast transcripts, articles) for relevant context.
                
                Use this tool when you need information about:
                - Player analysis and insights
                - Draft advice and strategy
                - Fantasy basketball tips and trends
                - Historical context about players or teams
                
                Args:
                    query: Search query to find relevant context
                    
                Returns:
                    Relevant context from the knowledge base
                """
                docs = self.retriever.invoke(query)
                if not docs:
                    return "No relevant context found in knowledge base."
                
                # Format results
                context_parts = []
                for i, doc in enumerate(docs, 1):
                    metadata = doc.metadata
                    source = metadata.get('url', metadata.get('source', 'Unknown'))
                    title = metadata.get('title', 'Untitled')
                    content = doc.page_content
                    
                    context_parts.append(f"[Source {i}: {title} ({source})]\n{content}")
                
                return "\n\n".join(context_parts)
            
            tools.append(search_knowledge_base)
        except Exception as e:
            logger.warning(f"RAG retrieval tool not available: {e}")
        
        return tools
    
    def invoke(self, query: str, **kwargs):
        """
        Invoke the agent with a query.
        
        Args:
            query: User query/question
            **kwargs: Additional arguments to pass to agent
            
        Returns:
            Agent response
        """
        inputs = {"messages": [("user", query)], **kwargs}
        return self.agent.invoke(inputs)
    
    def stream(self, query: str, **kwargs):
        """
        Stream agent responses.
        
        Args:
            query: User query/question
            **kwargs: Additional arguments to pass to agent
            
        Yields:
            Agent response events
        """
        inputs = {"messages": [("user", query)], **kwargs}
        for event in self.agent.stream(inputs):
            yield event
