"""
Base class for ClutchAI Agent tools.
"""

import json
from abc import ABC, abstractmethod
from typing import List
from langchain_core.tools import BaseTool as LangChainBaseTool


class ClutchAITool(ABC):
    """
    Base class for creating tool classes for LangChain agents.
    
    Subclasses should:
    1. Implement __init__ with their specific initialization
    2. Override _format_response if needed for custom formatting
    3. Implement get_all_tools to return a list of tool instances
    4. Create individual tool methods using @tool decorator
    """
    
    def _format_response(self, data) -> str:
        """
        Helper method to format API response data as JSON string.
        
        This is a base implementation that handles common cases.
        Subclasses can override this for API-specific formatting needs.
        
        Args:
            data: Response data from API call
            
        Returns:
            Formatted JSON string representation of the data
        """
        try:
            # Handle pandas DataFrames
            if hasattr(data, 'to_dict'):
                return json.dumps(data.to_dict('records'), default=str, indent=2)
            # Handle objects with get_dict() method (e.g., nba_api responses)
            elif hasattr(data, 'get_dict'):
                return json.dumps(data.get_dict(), default=str, indent=2)
            # Handle objects with get_json() method
            elif hasattr(data, 'get_json'):
                return data.get_json()
            # Handle objects with __dict__
            elif hasattr(data, '__dict__'):
                return json.dumps(data.__dict__, default=str, indent=2)
            # Handle dict/list
            elif isinstance(data, (dict, list)):
                return json.dumps(data, default=str, indent=2)
            else:
                return str(data)
        except Exception as e:
            return f"Error formatting response: {str(e)}\nRaw data: {str(data)}"
    
    @abstractmethod
    def get_all_tools(self) -> List[LangChainBaseTool]:
        """
        Get all available tools for this tool class.
        
        Returns:
            List of all LangChain tool instances
        """
        pass

