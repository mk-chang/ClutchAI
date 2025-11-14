"""
ClutchAI Agent - A ReACT AI Agent for Yahoo Fantasy Basketball League Management

This agent combines:
1. Yahoo Fantasy Sports API integration for live league data
2. Local ChromaDB vectorstore with Locked On Basketball Podcast transcripts for draft advice before using this agent.
"""

import os
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from yfpy.query import YahooFantasySportsQuery

from ClutchAI.rag.vectorstore import VectorstoreManager
from ClutchAI.tools.yahoofantasy import YahooFantasyTool
from ClutchAI.tools.nba_api import nbaAPITool

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
        # Store debug mode
        self.debug = debug
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
        
        # Create tools
        self.tools = self._create_tools()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=self.openai_api_key,
        )
        
        # Create agent
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt="You are a helpful assistant for a Yahoo Fantasy Sports league manager.",
        )
    
    def _initialize_vectorstore(self) -> Chroma:
        """
        Initialize and update ChromaDB vectorstore from local persistence.
        
        This method will:
        1. Check if vectorstore exists
        2. If not, create it by updating from YAML configuration
        3. If it exists, update it with any new resources from YAML
        4. Return the vectorstore instance
        
        Returns:
            Chroma vectorstore instance
            
        Raises:
            RuntimeError: If vectorstore creation/update fails
        """
        # Get path to vectordata.yaml (should be in rag directory)
        vectordata_yaml = Path(__file__).parent / "rag" / "vectordata.yaml"
        
        # Initialize VectorstoreManager to handle creation/updates
        vectorstore_manager = VectorstoreManager(
            vectordata_yaml=str(vectordata_yaml),
            chroma_persist_directory=self.chroma_persist_directory,
            openai_api_key=self.openai_api_key,
            env_file_location=self.env_file_location,
        )
        
        # Check if vectorstore exists
        existing_vectorstore = vectorstore_manager.get_vectorstore()
        vectorstore_exists = existing_vectorstore is not None
        
        # Update vectorstore (will create if it doesn't exist)
        try:
            print("Updating vectorstore from YAML configuration...")
            update_results = vectorstore_manager.update_vectorstore(
                chunk_size_seconds=30,
                skip_existing=True
            )
            
            # Print update summary
            if update_results['added'] > 0 or update_results['updated'] > 0:
                print(f"Vectorstore updated: {update_results['added']} added, "
                      f"{update_results['updated']} updated, "
                      f"{update_results['chunks_added']} chunks total")
            elif vectorstore_exists:
                print("Vectorstore is up to date.")
            else:
                print("Vectorstore created successfully.")
        except Exception as e:
            print(f"Warning: Error updating vectorstore: {e}")
            # Continue anyway - might be able to load existing vectorstore
        
        # Get the vectorstore instance
        vectorstore = vectorstore_manager.get_vectorstore()
        
        if vectorstore is None:
            # Try to create an empty vectorstore as fallback
            try:
                vectorstore = Chroma(
                    persist_directory=self.chroma_persist_directory,
                    embedding_function=OpenAIEmbeddings(api_key=self.openai_api_key),
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize ChromaDB vectorstore at {self.chroma_persist_directory}: {e}"
                ) from e
        
        # Verify vectorstore has documents
        try:
            doc_count = vectorstore._collection.count()
            if doc_count == 0:
                print("Warning: Vectorstore exists but is empty. It will be populated on next update.")
        except Exception as e:
            print(f"Warning: Could not verify vectorstore document count: {e}")
        
        return vectorstore
    
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
        
        - Vectorstore tool (1): retrieve contextual knowledge from vectorstore
        
        Returns:
            List of tool instances (45+ Yahoo Fantasy tools + 16 NBA API tools + 1 vectorstore tool)
        """
        # Create Yahoo Fantasy tools using YahooFantasyTool class
        # This includes all 45 Yahoo Fantasy Sports API tools
        yahoo_tool = YahooFantasyTool(self.query)
        yahoo_tools = yahoo_tool.get_all_tools()
        
        # Create NBA API tools using nbaAPITool class
        # This includes all 16 NBA API tools
        nba_tool = nbaAPITool()
        nba_tools = nba_tool.get_all_tools()
        
        @tool("vectorstore_retriever", description="Retrieve contextual knowledge from the vectorstore.")
        def retrieve_vectorstore(query: str) -> str:
            """Retrieve contextual knowledge from the vectorstore."""
            try:
                results = self.retriever.invoke(query)
                return "\n\n".join([r.page_content for r in results])
            except Exception as e:
                return f"Failed to retrieve knowledge: {e}"
        
        tools = yahoo_tools + nba_tools + [retrieve_vectorstore]
        return self._wrap_tools_with_debug_logging(tools)

    def _wrap_tools_with_debug_logging(self, tools):
        """Add terminal debug logging to each tool when debug mode is enabled."""
        if not self.debug:
            return tools

        wrapped_tools = []
        for tool in tools:
            tool_name = getattr(tool, "name", tool.__class__.__name__)
            updates = {}

            tool_func = getattr(tool, "func", None)
            if callable(tool_func):
                def func_wrapper(*args, _orig=tool_func, _name=tool_name, **kwargs):
                    print(f"🐛 [DEBUG] Tool '{_name}' called with args={args}, kwargs={kwargs}")
                    return _orig(*args, **kwargs)

                updates["func"] = func_wrapper

            tool_coroutine = getattr(tool, "coroutine", None)
            if callable(tool_coroutine):
                async def coroutine_wrapper(*args, _orig=tool_coroutine, _name=tool_name, **kwargs):
                    print(f"🐛 [DEBUG] Tool '{_name}' coroutine called with args={args}, kwargs={kwargs}")
                    return await _orig(*args, **kwargs)

                updates["coroutine"] = coroutine_wrapper

            if updates and hasattr(tool, "copy"):
                try:
                    tool = tool.copy(update=updates)
                except Exception as exc:  # pragma: no cover - debug logging only
                    print(f"🐛 [DEBUG] Failed to wrap tool '{tool_name}': {exc}")
            elif updates and self.debug:
                print(f"🐛 [DEBUG] Tool '{tool_name}' does not support copy(); skipping debug wrap")

            wrapped_tools.append(tool)

        return wrapped_tools
    
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

