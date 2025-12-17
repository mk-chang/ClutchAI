"""
Research Agent for Multi-Agent System

This agent is responsible for gathering data from all available sources:
- Yahoo Fantasy API
- NBA API
- RAG/Knowledge Base
- News/RSS Feeds
- Dynasty Rankings
"""

import os
import yaml
from pathlib import Path
from typing import Optional, List
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from yfpy.query import YahooFantasySportsQuery

from logger import get_logger
from agents.rag.rag_manager import RAGManager
from agents.tools.yahoo_api import YahooFantasyTool
from agents.tools.nba_api import nbaAPITool
from agents.tools.fantasy_news import FantasyNewsTool
from agents.tools.dynasty_ranking import DynastyRankingTool
from agents.tools.rotowire_rss import RotowireRSSFeedTool

logger = get_logger(__name__)


class ResearchAgent:
    """
    Research Agent that gathers data from multiple sources.
    """
    
    def __init__(
        self,
        query: YahooFantasySportsQuery,
        rag_manager: RAGManager,
        tools_config: dict,
        openai_api_key: Optional[str] = None,
        project_root: Optional[Path] = None,
        debug: bool = False,
    ):
        """
        Initialize Research Agent.
        
        Args:
            query: YahooFantasySportsQuery instance
            rag_manager: RAGManager instance for knowledge base access
            tools_config: Tools configuration dict (from tools_config.yaml)
            openai_api_key: OpenAI API key
            project_root: Path to project root for loading config
            debug: Enable debug logging
        """
        self.query = query
        self.rag_manager = rag_manager
        self.retriever = rag_manager.retriever
        self.tools_config = tools_config
        self.debug = debug
        self.logger = get_logger(__name__)
        
        # Load multi-agent config
        self.config = self._load_config(project_root)
        research_config = self.config.get('research', {})
        
        # Get model settings from config
        model_name = research_config.get('model_name', 'gpt-4o-mini')
        temperature = research_config.get('temperature', 0)
        
        # Get API key
        self.openai_api_key = openai_api_key or os.environ.get('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required.")
        
        # Create tools
        self.tools = self._create_tools()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=self.openai_api_key,
        )
        
        # Get system prompt from config
        system_prompt = research_config.get(
            'system_prompt',
            'You are a research specialist. Gather comprehensive data from all available sources.'
        )
        
        # Create agent
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt,
        )
    
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
    
    def _create_tools(self) -> List:
        """Create all research tools."""
        tools = []
        
        # Yahoo Fantasy API tools
        try:
            yahoo_tool = YahooFantasyTool(
                query=self.query,
                debug=self.debug,
            )
            tools.extend(yahoo_tool.get_all_tools())
            self.logger.debug("Yahoo Fantasy tools loaded")
        except Exception as e:
            self.logger.warning(f"Yahoo Fantasy tools not available: {e}")
        
        # NBA API tools
        try:
            nba_tool = nbaAPITool(debug=self.debug)
            tools.extend(nba_tool.get_all_tools())
            self.logger.debug("NBA API tools loaded")
        except Exception as e:
            self.logger.warning(f"NBA API tools not available: {e}")
        
        # RAG/Knowledge Base tool
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
            self.logger.debug("RAG/Knowledge Base tool loaded")
        except Exception as e:
            self.logger.warning(f"RAG retrieval tool not available: {e}")
        
        # Fantasy News tools
        try:
            news_urls = self.tools_config.get('yahoo_fantasy_news_urls', [])
            if news_urls:
                fantasy_news_tool = FantasyNewsTool(urls=news_urls, debug=self.debug)
                tools.extend(fantasy_news_tool.get_all_tools())
                self.logger.debug("Fantasy News tools loaded")
        except Exception as e:
            self.logger.warning(f"Fantasy News tools not available: {e}")
        
        # Dynasty Ranking tools
        try:
            dynasty_rankings_urls = self.tools_config.get('dynasty_rankings_url', [])
            dynasty_url = dynasty_rankings_urls[0] if dynasty_rankings_urls else None
            if dynasty_url:
                dynasty_tool = DynastyRankingTool(url=dynasty_url, debug=self.debug)
            else:
                dynasty_tool = DynastyRankingTool(debug=self.debug)
            tools.extend(dynasty_tool.get_all_tools())
            self.logger.debug("Dynasty Ranking tools loaded")
        except Exception as e:
            self.logger.warning(f"Dynasty Ranking tools not available: {e}")
        
        # Rotowire RSS tools
        try:
            rotowire_rss_url = self.tools_config.get('rotowire_rss_url')
            if rotowire_rss_url:
                rotowire_rss_tool = RotowireRSSFeedTool(rss_url=rotowire_rss_url, debug=self.debug)
                tools.extend(rotowire_rss_tool.get_all_tools())
                self.logger.debug("Rotowire RSS tools loaded")
        except Exception as e:
            self.logger.warning(f"Rotowire RSS tools not available: {e}")
        
        self.logger.info(f"Research Agent initialized with {len(tools)} tools")
        return tools
    
    def invoke(self, query: str, **kwargs):
        """
        Invoke the research agent with a query.
        
        Args:
            query: Research query/task
            **kwargs: Additional arguments
            
        Returns:
            Research data/response
        """
        self.logger.debug(f"Research Agent handling query: {query[:100]}...")
        
        inputs = {"messages": [("user", query)], **kwargs}
        response = self.agent.invoke(inputs)
        
        return response
    
    def stream(self, query: str, **kwargs):
        """Stream research agent responses."""
        inputs = {"messages": [("user", query)], **kwargs}
        for event in self.agent.stream(inputs):
            yield event

