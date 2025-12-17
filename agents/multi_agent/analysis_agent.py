"""
Analysis Agent for Multi-Agent System

This agent analyzes research data and generates recommendations with reasoning.
It uses only LLM (no tools) to focus on analysis and synthesis.
"""

import os
import yaml
from pathlib import Path
from typing import Optional
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from logger import get_logger

logger = get_logger(__name__)


class AnalysisAgent:
    """
    Analysis Agent that analyzes research data and generates recommendations.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        project_root: Optional[Path] = None,
        debug: bool = False,
    ):
        """
        Initialize Analysis Agent.
        
        Args:
            openai_api_key: OpenAI API key
            project_root: Path to project root for loading config
            debug: Enable debug logging
        """
        self.debug = debug
        self.logger = get_logger(__name__)
        
        # Load multi-agent config
        self.config = self._load_config(project_root)
        analysis_config = self.config.get('analysis', {})
        
        # Get model settings from config (default to gpt-4o for better reasoning)
        model_name = analysis_config.get('model_name', 'gpt-4o')
        temperature = analysis_config.get('temperature', 0)
        
        # Get API key
        self.openai_api_key = openai_api_key or os.environ.get('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required.")
        
        # Initialize LLM (no tools for analysis agent)
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=self.openai_api_key,
        )
        
        # Get system prompt from config
        system_prompt = analysis_config.get(
            'system_prompt',
            'You are an analysis specialist. Analyze research data and provide recommendations with reasoning.'
        )
        
        # Create agent with no tools (LLM only)
        self.agent = create_agent(
            model=self.llm,
            tools=[],  # No tools - pure analysis
            system_prompt=system_prompt,
        )
        
        self.logger.info("Analysis Agent initialized")
    
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
    
    def invoke(self, query: str, research_data: str, **kwargs):
        """
        Invoke the analysis agent with research data and original query.
        
        Args:
            query: Original user query
            research_data: Research data from Research Agent
            **kwargs: Additional arguments
            
        Returns:
            Analysis and recommendations
        """
        self.logger.debug(f"Analysis Agent analyzing query: {query[:100]}...")
        
        # Format the analysis task with research data
        analysis_prompt = f"""Original User Query: {query}

Research Data Gathered:
{research_data}

Please analyze the research data above and provide:
1. A comprehensive analysis of the findings
2. Clear, actionable recommendations
3. Reasoning for your recommendations
4. Supporting evidence from the research data
5. Any important caveats or considerations"""
        
        inputs = {"messages": [("user", analysis_prompt)], **kwargs}
        response = self.agent.invoke(inputs)
        
        return response
    
    def stream(self, query: str, research_data: str, **kwargs):
        """Stream analysis agent responses."""
        analysis_prompt = f"""Original User Query: {query}

Research Data Gathered:
{research_data}

Please analyze the research data above and provide recommendations with reasoning."""
        
        inputs = {"messages": [("user", analysis_prompt)], **kwargs}
        for event in self.agent.stream(inputs):
            yield event

