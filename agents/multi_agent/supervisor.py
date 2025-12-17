"""
Supervisor Agent for Multi-Agent System

This agent orchestrates the workflow between Research and Analysis agents.
It coordinates the sequential pipeline: Query → Research → Analysis → Response
"""

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from yfpy.query import YahooFantasySportsQuery

from logger import get_logger
from agents.multi_agent.research_agent import ResearchAgent
from agents.multi_agent.analysis_agent import AnalysisAgent
from agents.rag.rag_manager import RAGManager

logger = get_logger(__name__)


class SupervisorAgent:
    """
    Supervisor Agent that coordinates Research and Analysis agents.
    """
    
    def __init__(
        self,
        research_agent: ResearchAgent,
        analysis_agent: AnalysisAgent,
        openai_api_key: Optional[str] = None,
        project_root: Optional[Path] = None,
        debug: bool = False,
    ):
        """
        Initialize Supervisor Agent.
        
        Args:
            research_agent: ResearchAgent instance
            analysis_agent: AnalysisAgent instance
            openai_api_key: OpenAI API key
            project_root: Path to project root for loading config
            debug: Enable debug logging
        """
        self.research_agent = research_agent
        self.analysis_agent = analysis_agent
        self.debug = debug
        self.logger = get_logger(__name__)
        
        # Load multi-agent config
        self.config = self._load_config(project_root)
        supervisor_config = self.config.get('supervisor', {})
        
        # Get model settings from config (default to gpt-4o for coordination)
        model_name = supervisor_config.get('model_name', 'gpt-4o')
        temperature = supervisor_config.get('temperature', 0)
        
        # Get API key
        self.openai_api_key = openai_api_key or os.environ.get('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required.")
        
        # Create tools that delegate to Research and Analysis agents
        self.tools = self._create_tools()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=self.openai_api_key,
        )
        
        # Get system prompt from config
        system_prompt = supervisor_config.get(
            'system_prompt',
            'You are a supervisor agent. Coordinate research and analysis to answer user queries.'
        )
        
        # Create agent
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt,
        )
        
        self.logger.info("Supervisor Agent initialized")
    
    def _load_config(self, project_root: Optional[Path] = None) -> dict:
        """Load multi-agent configuration."""
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent.parent
        
        config_path = project_root / 'config' / 'multi_agent_config.yaml'
        
        if not config_path.exists():
            self.logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return {}
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            self.logger.warning(f"Error loading multi_agent_config.yaml: {e}. Using defaults.")
            return {}
    
    def _create_tools(self) -> list:
        """Create tools that delegate to Research and Analysis agents."""
        tools = []
        
        # Capture agent references for use in tool closures
        research_agent = self.research_agent
        analysis_agent = self.analysis_agent
        logger = self.logger
        
        @tool
        def call_research_agent(query: str) -> str:
            """
            Call the Research Agent to gather data from all available sources.
            
            Use this tool first when you receive a user query. The Research Agent will
            gather comprehensive data from Yahoo API, NBA API, knowledge base, news feeds, etc.
            
            Args:
                query: The research task/query to send to the Research Agent
                
            Returns:
                Research data gathered from all sources
            """
            try:
                response = research_agent.invoke(query)
                
                # Extract response text
                if isinstance(response, dict):
                    if "output" in response:
                        return str(response["output"])
                    elif "messages" in response:
                        last_msg = response["messages"][-1]
                        if hasattr(last_msg, "content"):
                            return str(last_msg.content)
                        elif isinstance(last_msg, tuple):
                            return str(last_msg[1])
                        else:
                            return str(last_msg)
                    else:
                        return str(response)
                elif isinstance(response, list) and len(response) > 0:
                    last_msg = response[-1]
                    if hasattr(last_msg, "content"):
                        return str(last_msg.content)
                    elif isinstance(last_msg, tuple):
                        return str(last_msg[1])
                    else:
                        return str(last_msg)
                else:
                    return str(response)
            except Exception as e:
                logger.error(f"Error in Research Agent: {e}")
                return f"Error gathering research data: {str(e)}"
        
        @tool
        def call_analysis_agent(query: str, research_data: str) -> str:
            """
            Call the Analysis Agent to analyze research data and generate recommendations.
            
            Use this tool after gathering research data. The Analysis Agent will analyze
            the research data and provide recommendations with reasoning.
            
            Args:
                query: The original user query
                research_data: The research data gathered by the Research Agent
                
            Returns:
                Analysis and recommendations
            """
            try:
                response = analysis_agent.invoke(query, research_data)
                
                # Extract response text
                if isinstance(response, dict):
                    if "output" in response:
                        return str(response["output"])
                    elif "messages" in response:
                        last_msg = response["messages"][-1]
                        if hasattr(last_msg, "content"):
                            return str(last_msg.content)
                        elif isinstance(last_msg, tuple):
                            return str(last_msg[1])
                        else:
                            return str(last_msg)
                    else:
                        return str(response)
                elif isinstance(response, list) and len(response) > 0:
                    last_msg = response[-1]
                    if hasattr(last_msg, "content"):
                        return str(last_msg.content)
                    elif isinstance(last_msg, tuple):
                        return str(last_msg[1])
                    else:
                        return str(last_msg)
                else:
                    return str(response)
            except Exception as e:
                logger.error(f"Error in Analysis Agent: {e}")
                return f"Error analyzing data: {str(e)}"
        
        tools.extend([call_research_agent, call_analysis_agent])
        return tools
    
    def invoke(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Invoke supervisor to handle query through Research → Analysis pipeline.
        
        Args:
            query: User query
            **kwargs: Additional arguments
            
        Returns:
            Supervisor response
        """
        self.logger.debug(f"Supervisor handling query: {query[:100]}...")
        
        inputs = {"messages": [("user", query)], **kwargs}
        response = self.agent.invoke(inputs)
        
        return response
    
    def stream(self, query: str, **kwargs):
        """Stream supervisor responses."""
        inputs = {"messages": [("user", query)], **kwargs}
        for event in self.agent.stream(inputs):
            yield event

