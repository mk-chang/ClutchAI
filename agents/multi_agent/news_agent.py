"""
News Agent for Multi-Agent System

This agent specializes in gathering news, insights, and contextual information:
- Fantasy news from Yahoo and other sources
- RSS feeds (Rotowire, etc.)
- Knowledge base (podcast transcripts, articles)
- Dynasty rankings
"""

from pathlib import Path
from typing import Optional, List
from langchain_core.tools import tool

from agents.multi_agent.base_agent import BaseAgent
from agents.tools.fantasy_news import FantasyNewsTool
from agents.tools.dynasty_ranking import DynastyRankingTool
from agents.tools.rotowire_rss import RotowireRSSFeedTool

logger = None  # Will be set by BaseAgent


class NewsAgent(BaseAgent):
    """
    News Agent that specializes in news, RSS feeds, knowledge base, and rankings.
    """
    
    def _get_config_section(self) -> str:
        """Get the configuration section name for this agent."""
        return 'news'
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt if not found in config."""
        return """You are a news and insights specialist agent for fantasy basketball analysis.
Your role is to gather information from news sources, expert insights, and rankings:
- Fantasy news: Latest updates from Yahoo Fantasy and other sources
- RSS feeds: Injury updates, breaking news from Rotowire and other feeds
- Knowledge base: Expert insights from podcast transcripts and articles
- Dynasty rankings: Long-term player value rankings

When given a research task, use all relevant tools to gather news, insights, and contextual information.
Be thorough and provide structured data that can be easily analyzed."""
    
    def _create_tools(self) -> List:
        """Create news, RSS, knowledge base, and ranking tools."""
        # Start with base tools (includes BasicTool)
        tools = list(super()._create_base_tools())
        
        # RAG/Knowledge Base tool
        if self.retriever is None:
            self.logger.warning("RAG retriever not available. Knowledge base tool will not be available.")
        else:
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
        
        self.logger.info(f"News Agent initialized with {len(tools)} tools")
        return tools

