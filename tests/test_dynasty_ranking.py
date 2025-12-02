"""
Integration tests for DynastyRankingTool that test actual website connections.

These tests require:
- Network connectivity
- Real Hashtag Basketball website access

Run with: pytest -m integration tests/test_dynasty_ranking.py
"""

import pytest
from pathlib import Path
import yaml

from ClutchAI.tools.dynasty_ranking import DynastyRankingTool


@pytest.fixture
def dynasty_rankings_url():
    """Load dynasty rankings URL from agent_config.yaml."""
    config_path = Path(__file__).parent.parent / "ClutchAI" / "agent_config.yaml"
    
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            
            urls = config.get('dynasty_rankings_url', [])
            if urls:
                # Return first URL from list for testing
                if isinstance(urls, list) and len(urls) > 0:
                    return urls[0]
                elif isinstance(urls, str):
                    return urls
        except Exception:
            pass
    
    # Fallback to default URL if config not found or doesn't have the key
    return "https://hashtagbasketball.com/fantasy-basketball-dynasty-rankings"


@pytest.mark.integration
class TestDynastyRankingToolIntegration:
    """Integration tests for DynastyRankingTool with real URL from agent_config.yaml.
    
    These tests make actual network calls to the Hashtag Basketball dynasty rankings URL.
    They require network connectivity and may be slower.
    
    Run with: pytest -m integration tests/test_dynasty_ranking.py::TestDynastyRankingToolIntegration
    """
    
    @pytest.fixture(autouse=True)
    def setup(self, dynasty_rankings_url):
        """Setup for integration tests using URL from config."""
        self.url = dynasty_rankings_url
        self.tool = DynastyRankingTool(url=self.url, cache_duration_hours=1)
        self.tools = self.tool.get_all_tools()
        yield
    
    @pytest.mark.integration
    def test_get_player_dynasty_rank_tool_integration(self, save_test_output):
        """Test get_player_dynasty_rank tool with real URL from config.
        
        This test actually calls the tool with a real player name and scrapes
        the rankings from the configured URL.
        Sample output saved to: tests/test_outputs/test_get_player_dynasty_rank_tool_integration_get_player_dynasty_rank.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_player_dynasty_rank')
        
        # Test with a well-known player
        test_player = "Luka Doncic"
        
        # Call real tool
        result = get_tool.invoke({"player_name": test_player})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_player_dynasty_rank", result)
        
        # Verify the result
        assert "Player dynasty ranking data" in result or "not found" in result.lower()
        assert "Failed" not in result
        assert len(result) > 50  # Should have substantial content
    
    @pytest.mark.integration
    def test_get_all_dynasty_rankings_tool_integration(self, save_test_output):
        """Test get_all_dynasty_rankings tool with real URL from config.
        
        This test actually gets all dynasty rankings from the configured URL.
        Sample output saved to: tests/test_outputs/test_get_all_dynasty_rankings_tool_integration_get_all_dynasty_rankings.txt
        """
        get_all_tool = next(t for t in self.tools if t.name == 'get_all_dynasty_rankings')
        
        # Call real tool
        result = get_all_tool.invoke({})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_all_dynasty_rankings", result)
        
        # Verify the result
        assert "All dynasty rankings" in result
        assert "players" in result.lower()
        assert "Failed" not in result
        assert len(result) > 100  # Should have substantial content
    
    @pytest.mark.integration
    def test_get_top_dynasty_rankings_tool_integration(self, save_test_output):
        """Test get_top_dynasty_rankings tool with real URL from config.
        
        This test actually gets top dynasty rankings from the configured URL.
        Sample output saved to: tests/test_outputs/test_get_top_dynasty_rankings_tool_integration_get_top_dynasty_rankings.txt
        """
        get_top_tool = next(t for t in self.tools if t.name == 'get_top_dynasty_rankings')
        
        # Test with top 10 players
        top_n = 10
        
        # Call real tool
        result = get_top_tool.invoke({"top_n": top_n})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_top_dynasty_rankings", result)
        
        # Verify the result
        assert "Top" in result and "dynasty rankings" in result.lower()
        assert "Failed" not in result
        assert len(result) > 50  # Should have substantial content
    
    @pytest.mark.integration
    def test_get_dynasty_rankings_by_position_tool_integration(self, save_test_output):
        """Test get_dynasty_rankings_by_position tool with real URL from config.
        
        This test actually gets dynasty rankings filtered by position from the configured URL.
        Sample output saved to: tests/test_outputs/test_get_dynasty_rankings_by_position_tool_integration_get_dynasty_rankings_by_position.txt
        """
        get_by_position_tool = next(t for t in self.tools if t.name == 'get_dynasty_rankings_by_position')
        
        # Test with a common position
        position = "PG"
        
        # Call real tool
        result = get_by_position_tool.invoke({"position": position})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_dynasty_rankings_by_position", result)
        
        # Verify the result
        assert "Found" in result or "No players found" in result
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content