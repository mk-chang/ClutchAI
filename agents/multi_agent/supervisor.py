"""
Supervisor Agent for Multi-Agent System

This agent orchestrates the workflow between specialized Research agents and Analysis agent.
It coordinates the pipeline: Query → Specialized Research Agents → Analysis → Response
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
from langchain_core.tools import tool

from agents.multi_agent.base_agent import BaseAgent
from agents.multi_agent.yahoo_fantasy_agent import YahooFantasyAgent
from agents.multi_agent.statistic_agent import StatisticAgent
from agents.multi_agent.news_agent import NewsAgent
from agents.multi_agent.analyst_agent import FantasyAnalystAgent

logger = None  # Will be set by BaseAgent


class SupervisorAgent(BaseAgent):
    """
    Supervisor Agent that coordinates specialized Research agents and Analysis agent.
    """
    
    def __init__(
        self,
        yahoo_fantasy_agent: YahooFantasyAgent,
        statistic_agent: StatisticAgent,
        news_agent: NewsAgent,
        analysis_agent: FantasyAnalystAgent,
        user_context: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        project_root: Optional[Path] = None,
        debug: bool = False,
    ):
        """
        Initialize Supervisor Agent.
        
        Args:
            yahoo_fantasy_agent: YahooFantasyAgent instance
            statistic_agent: StatisticAgent instance
            news_agent: NewsAgent instance
            analysis_agent: FantasyAnalystAgent instance
            user_context: Pre-gathered context about user's teams, leagues, and rosters
            openai_api_key: OpenAI API key
            project_root: Path to project root for loading config
            debug: Enable debug logging
        """
        # Store agent-specific dependencies before calling super().__init__
        self.yahoo_fantasy_agent = yahoo_fantasy_agent
        self.statistic_agent = statistic_agent
        self.news_agent = news_agent
        self.analysis_agent = analysis_agent
        
        # Call parent initializer (which handles user_context)
        super().__init__(
            user_context=user_context,
            openai_api_key=openai_api_key,
            project_root=project_root,
            debug=debug,
        )
    
    def _get_config_section(self) -> str:
        """Get the configuration section name for this agent."""
        return 'supervisor'
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt if not found in config."""
        return 'You are a supervisor agent. Coordinate research and analysis to answer user queries.'
    
    def _create_tools(self) -> List:
        """Create tools that delegate to specialized Research agents and Analysis agent."""
        # Start with base tools (includes BasicTool)
        tools = list(super()._create_base_tools())
        
        # Capture agent references for use in tool closures
        yahoo_fantasy_agent = self.yahoo_fantasy_agent
        statistic_agent = self.statistic_agent
        news_agent = self.news_agent
        analysis_agent = self.analysis_agent
        logger = self.logger
        
        def _extract_response_text(response) -> str:
            """Helper to extract text from agent response."""
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
        
        @tool
        def call_yahoo_fantasy_agent(query: str) -> str:
            """
            Call the Yahoo Fantasy Agent to gather Yahoo Fantasy API data.
            
            Use this tool for Yahoo Fantasy-specific queries:
            - League data: standings, settings, teams, players
            - Team data: rosters, stats, matchups, draft results
            - Player data: stats, ownership, draft analysis
            - Transaction data: trades, waiver claims, adds/drops
            
            Args:
                query: The research task/query to send to the Yahoo Fantasy Agent
                
            Returns:
                Yahoo Fantasy data gathered from the API
            """
            try:
                response = yahoo_fantasy_agent.invoke(query)
                return _extract_response_text(response)
            except Exception as e:
                logger.error(f"Error in Yahoo Fantasy Agent: {e}")
                return f"Error gathering Yahoo Fantasy data: {str(e)}"
        
        @tool
        def call_statistic_agent(query: str) -> str:
            """
            Call the Statistic Agent to gather NBA statistics and game data.
            
            Use this tool for NBA statistics queries:
            - Player statistics: current season, career, game logs, splits
            - Team statistics: team stats, game logs, performance metrics
            - Game data: scoreboards, live scores, game information
            - Player and team information: rosters, details, metadata
            
            Args:
                query: The research task/query to send to the Statistic Agent
                
            Returns:
                NBA statistics and game data
            """
            try:
                response = statistic_agent.invoke(query)
                return _extract_response_text(response)
            except Exception as e:
                logger.error(f"Error in Statistic Agent: {e}")
                return f"Error gathering statistics data: {str(e)}"
        
        @tool
        def call_news_agent(query: str) -> str:
            """
            Call the News Agent to gather news, insights, and contextual information.
            
            Use this tool for news and insights queries:
            - Fantasy news: Latest updates from Yahoo Fantasy and other sources
            - RSS feeds: Injury updates, breaking news from Rotowire and other feeds
            - Knowledge base: Expert insights from podcast transcripts and articles
            - Dynasty rankings: Long-term player value rankings
            
            Args:
                query: The research task/query to send to the News Agent
                
            Returns:
                News, insights, and contextual information
            """
            try:
                response = news_agent.invoke(query)
                return _extract_response_text(response)
            except Exception as e:
                logger.error(f"Error in News Agent: {e}")
                return f"Error gathering news data: {str(e)}"
        
        @tool
        def call_analysis_agent(query: str, research_data: str) -> str:
            """
            Call the Analysis Agent to analyze research data and generate recommendations.
            
            Use this tool after gathering research data from one or more research agents.
            The Analysis Agent will analyze the research data and provide recommendations with reasoning.
            
            Args:
                query: The original user query
                research_data: The research data gathered by one or more research agents
                
            Returns:
                Analysis and recommendations
            """
            try:
                response = analysis_agent.invoke(query, research_data)
                return _extract_response_text(response)
            except Exception as e:
                logger.error(f"Error in Analysis Agent: {e}")
                return f"Error analyzing data: {str(e)}"
        
        tools.extend([
            call_yahoo_fantasy_agent,
            call_statistic_agent,
            call_news_agent,
            call_analysis_agent
        ])
        return tools

