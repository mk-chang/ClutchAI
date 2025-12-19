"""
Research Agent for Multi-Agent System

This agent is responsible for gathering data from all available sources:
- Yahoo Fantasy API
- NBA API
- RAG/Knowledge Base
- News/RSS Feeds
- Dynasty Rankings
"""

from pathlib import Path
from typing import Optional, List
from langchain_core.tools import tool
from yfpy.query import YahooFantasySportsQuery

from agents.multi_agent.base_agent import BaseAgent
from agents.rag.rag_manager import RAGManager
from agents.tools.yahoo_api import YahooFantasyTool
from agents.tools.nba_api import nbaAPITool
from agents.tools.fantasy_news import FantasyNewsTool
from agents.tools.dynasty_ranking import DynastyRankingTool
from agents.tools.rotowire_rss import RotowireRSSFeedTool

logger = None  # Will be set by BaseAgent


class ResearchAgent(BaseAgent):
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
        # Call parent initializer (which stores query, rag_manager, tools_config)
        super().__init__(
            query=query,
            rag_manager=rag_manager,
            tools_config=tools_config,
            openai_api_key=openai_api_key,
            project_root=project_root,
            debug=debug,
        )
    
    def _get_config_section(self) -> str:
        """Get the configuration section name for this agent."""
        return 'research'
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt if not found in config."""
        return 'You are a research specialist. Gather comprehensive data from all available sources.'
    
    def _create_tools(self) -> List:
        """Create all research tools."""
        # Start with base tools (includes BasicTool)
        tools = list(super()._create_base_tools())
        
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

