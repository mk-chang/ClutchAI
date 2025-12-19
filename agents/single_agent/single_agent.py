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

import tiktoken

from langchain_core.tools import tool, StructuredTool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from yfpy.query import YahooFantasySportsQuery

from logger import get_logger, setup_logging
from agents.rag.rag_manager import RAGManager
from agents.tools.yahoo_api import YahooFantasyTool
from agents.tools.nba_api import nbaAPITool
from agents.tools.fantasy_news import FantasyNewsTool
from agents.tools.dynasty_ranking import DynastyRankingTool
from agents.tools.rotowire_rss import RotowireRSSFeedTool
from agents.tools.basic import BasicTool
from data.cloud_sql.connection import PostgresConnection
from data.cloud_sql.schema import get_default_table_name

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
        table_name: Optional[str] = None,
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
            table_name: Name of the vector table in PostgreSQL (defaults to env var CLOUDSQL_VECTOR_TABLE)
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
        # RAGManager will load config from config/rag_config.yaml automatically
        self.table_name = table_name or get_default_table_name()
        self.rag_manager = RAGManager(
            connection=connection,
            table_name=self.table_name,
            openai_api_key=self.openai_api_key,
            project_root=self.env_file_location,
        )
        
        # Initialize vectorstore and retriever
        self.vectorstore = self._initialize_vectorstore()
        self.retriever = self.rag_manager.retriever
        
        # Load agent configuration
        self.agent_config = self._load_agent_config()
        # Store max_tokens from config (default: 150000)
        self.max_tokens = self.agent_config.get('max_tokens', 150000)
        
        # Load tools configuration
        self.tools_config = self._load_tools_config()
        
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
            logger.debug(f"  Table: {stats.get('table_name', self.table_name)}")
            logger.debug(f"  Total documents: {stats.get('row_count', 0):,}")
            logger.debug(f"  Unique resources: {stats.get('unique_resources', 0)}")
            
            # Display source type distribution
            source_types = stats.get('source_types', {})
            if source_types:
                logger.debug(f"  Source types:")
                for source_type, count in source_types.items():
                    logger.debug(f"    - {source_type}: {count} documents")
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
        # agent_config.yaml is in config/ directory
        config_path = Path(__file__).parent.parent.parent / "config" / "agent_config.yaml"
        
        if not config_path.exists():
            # Return default config if file doesn't exist
            return {
                'system_prompt': 'You are a helpful assistant for a Yahoo Fantasy Sports league manager.'
            }
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            
            # Ensure required keys exist with defaults
            if 'system_prompt' not in config:
                config['system_prompt'] = 'You are a helpful assistant for a Yahoo Fantasy Sports league manager.'
            if 'max_tokens' not in config:
                config['max_tokens'] = 150000
            
            return config
        except Exception as e:
            logger.warning(f"Error loading agent_config.yaml: {e}. Using defaults.")
            return {
                'system_prompt': 'You are a helpful assistant for a Yahoo Fantasy Sports league manager.'
            }
    
    def _load_tools_config(self) -> dict:
        """
        Load tools configuration from tools_config.yaml.
        
        Returns:
            Dictionary with tools configuration
        """
        # tools_config.yaml is in config/ directory
        config_path = Path(__file__).parent.parent.parent / "config" / "tools_config.yaml"
        
        if not config_path.exists():
            # Return default config if file doesn't exist
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
                debug=self.debug,
            )
            tools.extend(yahoo_tool.get_all_tools())
        except Exception as e:
            logger.warning(f"Yahoo Fantasy tools not available: {e}")
        
        # NBA API tools
        try:
            nba_tool = nbaAPITool()
            tools.extend(nba_tool.get_all_tools())
        except Exception as e:
            logger.warning(f"NBA API tools not available: {e}")
        
        # Fantasy News tools
        try:
            news_urls = self.tools_config.get('yahoo_fantasy_news_urls', [])
            if news_urls:
                fantasy_news_tool = FantasyNewsTool(urls=news_urls)
                tools.extend(fantasy_news_tool.get_all_tools())
        except Exception as e:
            logger.warning(f"Fantasy News tools not available: {e}")
        
        # Dynasty Ranking tools
        try:
            dynasty_rankings_urls = self.tools_config.get('dynasty_rankings_url', [])
            # Use first URL if list provided, otherwise use default
            dynasty_url = dynasty_rankings_urls[0] if dynasty_rankings_urls else None
            if dynasty_url:
                dynasty_tool = DynastyRankingTool(url=dynasty_url)
            else:
                dynasty_tool = DynastyRankingTool()
            tools.extend(dynasty_tool.get_all_tools())
        except Exception as e:
            logger.warning(f"Dynasty Ranking tools not available: {e}")
        
        # Rotowire RSS tools
        try:
            rotowire_rss_url = self.tools_config.get('rotowire_rss_url')
            if rotowire_rss_url:
                rotowire_rss_tool = RotowireRSSFeedTool(rss_url=rotowire_rss_url)
                tools.extend(rotowire_rss_tool.get_all_tools())
        except Exception as e:
            logger.warning(f"Rotowire RSS tools not available: {e}")
        
        # Basic utility tools (date/time)
        try:
            basic_tool = BasicTool(debug=self.debug)
            tools.extend(basic_tool.get_all_tools())
        except Exception as e:
            logger.warning(f"Basic tools not available: {e}")
        
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
        Chat with the agent, supporting conversation history.
        
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
        
        # Invoke agent with conversation history
        inputs = {"messages": messages}
        response = self.agent.invoke(inputs)
        
        # Extract text from response
        # LangChain create_agent typically returns a dict with 'output' or 'messages' key
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
