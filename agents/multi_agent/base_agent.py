"""
Base Agent Class for Multi-Agent System

This base class provides common functionality for all agents in the multi-agent system:
- Configuration loading
- LLM initialization
- Agent creation
- Common invoke/stream methods
"""

import os
import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Any

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from yfpy.query import YahooFantasySportsQuery

from logger import get_logger
from agents.tools.basic import BasicTool
from agents.rag.rag_manager import RAGManager


class BaseAgent(ABC):
    """
    Base class for all agents in the multi-agent system.
    
    Provides common functionality for:
    - Configuration loading
    - LLM initialization
    - Agent creation
    - Invoke and stream methods
    """
    
    def __init__(
        self,
        query: Optional[YahooFantasySportsQuery] = None,
        rag_manager: Optional[RAGManager] = None,
        tools_config: Optional[dict] = None,
        user_context: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        project_root: Optional[Path] = None,
        debug: bool = False,
    ):
        """
        Initialize Base Agent.
        
        Args:
            query: YahooFantasySportsQuery instance (optional, for Yahoo-specific agents)
            rag_manager: RAGManager instance for knowledge base access (optional)
            tools_config: Tools configuration dict (from tools_config.yaml) (optional)
            user_context: Pre-gathered context about user's teams, leagues, and rosters (optional)
            openai_api_key: OpenAI API key
            project_root: Path to project root for loading config
            debug: Enable debug logging
        """
        # Store agent-specific dependencies (for research agents)
        self.query = query
        self.rag_manager = rag_manager
        self.retriever = rag_manager.retriever if rag_manager else None
        self.tools_config = tools_config or {}
        self.user_context = user_context or ""
        
        self.debug = debug
        self.logger = get_logger(self.__class__.__name__)
        
        # Load multi-agent config
        self.config = self._load_config(project_root)
        agent_config = self.config.get(self._get_config_section(), {})
        
        # Get model settings from config
        model_name = agent_config.get('model_name', 'gpt-4o-mini')
        temperature = agent_config.get('temperature', 0)
        
        # Get API key
        self.openai_api_key = openai_api_key or os.environ.get('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required.")
        
        # Create tools (implemented by subclasses)
        self.tools = self._create_tools()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=self.openai_api_key,
        )
        
        # Get system prompt from config
        system_prompt = agent_config.get(
            'system_prompt',
            self._get_default_system_prompt()
        )
        
        # Enhance system prompt with user context if available
        system_prompt = self._enhance_system_prompt(system_prompt)
        
        # Create agent
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt,
        )
        
        self.logger.info(f"{self.__class__.__name__} initialized")
    
    def _load_config(self, project_root: Optional[Path] = None) -> dict:
        """
        Load multi-agent configuration.
        
        Args:
            project_root: Path to project root
            
        Returns:
            Configuration dictionary
        """
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent.parent
        
        config_path = project_root / 'config' / 'multiagent_config.yaml'
        
        if not config_path.exists():
            self.logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return {}
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            self.logger.warning(f"Error loading multiagent_config.yaml: {e}. Using defaults.")
            return {}
    
    @abstractmethod
    def _get_config_section(self) -> str:
        """
        Get the configuration section name for this agent.
        
        Returns:
            Configuration section name (e.g., 'research', 'analysis', 'supervisor')
        """
        pass
    
    def _create_base_tools(self) -> List:
        """
        Create base utility tools available to all agents.
        Subclasses can override this to customize or disable base tools.
        
        Returns:
            List of base tools (e.g., BasicTool utilities)
        """
        tools = []
        
        # Basic utility tools (date/time) - available to all agents
        try:
            basic_tool = BasicTool(debug=self.debug)
            tools.extend(basic_tool.get_all_tools())
            self.logger.debug("Basic utility tools loaded")
        except Exception as e:
            self.logger.warning(f"Basic tools not available: {e}")
        
        return tools
    
    @abstractmethod
    def _create_tools(self) -> List:
        """
        Create tools for this agent.
        
        Subclasses must implement this method to provide their specific tools.
        To include base tools (e.g., BasicTool), call self._create_base_tools() and extend
        the result with agent-specific tools. To exclude base tools, return only agent-specific tools.
        
        Returns:
            List of tools for the agent
        """
        pass
    
    def _get_default_system_prompt(self) -> str:
        """
        Get default system prompt if not found in config.
        
        Subclasses can override this to provide a default prompt.
        
        Returns:
            Default system prompt string
        """
        return f"You are a {self.__class__.__name__}. Perform your assigned tasks effectively."
    
    def _enhance_system_prompt(self, base_prompt: str) -> str:
        """
        Enhance system prompt with user context if available.
        
        Subclasses can override this to customize how context is added.
        
        Args:
            base_prompt: Base system prompt from config or default
            
        Returns:
            Enhanced system prompt with user context appended (if available)
        """
        if self.user_context:
            return f"{base_prompt}\n\n=== USER CONTEXT ===\n{self.user_context}\n"
        return base_prompt
    
    def invoke(self, query: str, **kwargs) -> Any:
        """
        Invoke the agent with a query.
        
        Args:
            query: Query/task for the agent
            **kwargs: Additional arguments
            
        Returns:
            Agent response
        """
        self.logger.debug(f"{self.__class__.__name__} handling query: {query[:100]}...")
        
        inputs = {"messages": [("user", query)], **kwargs}
        response = self.agent.invoke(inputs)
        
        return response
    
    def stream(self, query: str, **kwargs):
        """
        Stream agent responses.
        
        Args:
            query: Query/task for the agent
            **kwargs: Additional arguments
            
        Yields:
            Agent response events
        """
        inputs = {"messages": [("user", query)], **kwargs}
        for event in self.agent.stream(inputs):
            yield event

