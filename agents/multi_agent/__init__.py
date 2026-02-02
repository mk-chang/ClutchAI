"""
Multi-Agent System for ClutchAI

This module provides a multi-agent system with:
- Supervisor Agent: Orchestrates workflow
- Yahoo Fantasy Agent: Gathers Yahoo Fantasy API data
- Statistic Agent: Gathers NBA statistics and game data
- News Agent: Gathers news, insights, and contextual information
- Analysis Agent: Analyzes data and generates recommendations
"""

from agents.multi_agent.multi_agent_system import MultiAgentSystem

__all__ = ['MultiAgentSystem']

