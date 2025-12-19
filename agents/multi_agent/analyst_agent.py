"""
Analysis Agent for Multi-Agent System

This agent analyzes research data and generates recommendations with reasoning.
It uses RAG to access analyst insights from podcast transcripts and articles.
"""

from pathlib import Path
from typing import Optional, List
from langchain_core.tools import tool

from agents.multi_agent.base_agent import BaseAgent
from agents.rag.rag_manager import RAGManager

logger = None  # Will be set by BaseAgent


class FantasyAnalystAgent(BaseAgent):
    """
    Analysis Agent that analyzes research data and generates recommendations.
    Has access to RAG for analyst insights from podcasts and articles.
    """
    
    def __init__(
        self,
        rag_manager: Optional[RAGManager] = None,
        user_context: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        project_root: Optional[Path] = None,
        debug: bool = False,
    ):
        """
        Initialize Analysis Agent.
        
        Args:
            rag_manager: RAGManager instance for knowledge base access (optional)
            user_context: Pre-gathered context about user's teams, leagues, and rosters (optional)
            openai_api_key: OpenAI API key
            project_root: Path to project root for loading config
            debug: Enable debug logging
        """
        # Store RAG manager before calling super().__init__
        self.rag_manager = rag_manager
        self.retriever = rag_manager.retriever if rag_manager else None
        
        # Call parent initializer (which handles user_context)
        super().__init__(
            rag_manager=rag_manager,
            user_context=user_context,
            openai_api_key=openai_api_key,
            project_root=project_root,
            debug=debug,
        )
    
    def _get_config_section(self) -> str:
        """Get the configuration section name for this agent."""
        return 'analysis'
    
    def _create_tools(self) -> List:
        """Create tools for this agent (includes RAG if available)."""
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
                    Search the knowledge base (podcast transcripts, articles) for relevant analyst insights.
                    
                    Use this tool when you need:
                    - Expert analyst opinions and takes on players
                    - Draft strategy and fantasy basketball insights
                    - Historical context and trends from analyst commentary
                    - Specific analyst recommendations and reasoning
                    
                    IMPORTANT: When multiple sources are found, prioritize and reference the most recent content first.
                    Analyst opinions and player situations change over time, so recent insights are typically more relevant.
                    
                    Args:
                        query: Search query to find relevant analyst insights
                        
                    Returns:
                        Relevant context from analyst podcasts and articles, with most recent content prioritized
                    """
                    docs = self.retriever.invoke(query)
                    if not docs:
                        return "No relevant analyst insights found in knowledge base."
                    
                    # Format results (docs are typically returned in relevance order, but we'll note recency)
                    context_parts = []
                    for i, doc in enumerate(docs, 1):
                        metadata = doc.metadata
                        source = metadata.get('url', metadata.get('source', 'Unknown'))
                        title = metadata.get('title', 'Untitled')
                        # Check for date metadata to help prioritize
                        date = metadata.get('date', metadata.get('published_date', metadata.get('created_at', '')))
                        content = doc.page_content
                        
                        date_str = f" (Date: {date})" if date else ""
                        context_parts.append(f"[Source {i}: {title} ({source}){date_str}]\n{content}")
                    
                    return "\n\n".join(context_parts)
                
                tools.append(search_knowledge_base)
                self.logger.debug("RAG/Knowledge Base tool loaded for Analysis Agent")
            except Exception as e:
                self.logger.warning(f"RAG retrieval tool not available: {e}")
        
        return tools
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt if not found in config."""
        return 'You are an analysis specialist. Analyze research data and provide recommendations with reasoning.'
    
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
        
        # Call parent invoke with the formatted prompt
        return super().invoke(analysis_prompt, **kwargs)
    
    def stream(self, query: str, research_data: str, **kwargs):
        """Stream analysis agent responses."""
        analysis_prompt = f"""Original User Query: {query}

Research Data Gathered:
{research_data}

Please analyze the research data above and provide recommendations with reasoning."""
        
        # Call parent stream with the formatted prompt
        yield from super().stream(analysis_prompt, **kwargs)

