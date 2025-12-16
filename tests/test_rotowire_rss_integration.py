"""
Integration tests for RotowireRSSFeedTool that test actual RSS feed parsing.

These tests require:
- Network connectivity
- Real Rotowire RSS feed access

Run with: pytest -m integration tests/test_rotowire_rss_integration.py
"""

import pytest
from pathlib import Path
import yaml

from agents.tools.rotowire_rss import RotowireRSSFeedTool


def _load_test_config():
    """Load test configuration from test_config.yaml file."""
    config_path = Path(__file__).parent / 'test_config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        return config.get('rotowire_rss', {})
    return {}


@pytest.fixture
def rotowire_rss_url():
    """Load Rotowire RSS URL from agent_config.yaml."""
    config_path = Path(__file__).parent.parent / "agents" / "agent_config.yaml"
    
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            
            url = config.get('rotowire_rss_url')
            if url:
                return url
        except Exception:
            pass
    
    # Fallback to default URL if config not found
    return "https://www.rotowire.com/rss/news.php?sport=NBA"


@pytest.mark.integration
class TestRotowireRSSFeedToolIntegration:
    """Integration tests for RotowireRSSFeedTool with real RSS feed.
    
    These tests make actual network calls to the Rotowire RSS feed.
    They require network connectivity and may be slower.
    
    Run with: pytest -m integration tests/test_rotowire_rss_integration.py
    """
    
    # Load test configuration from YAML file
    _test_config = _load_test_config()
    
    # Test limit
    # Configured in tests/test_config.yaml
    TEST_LIMIT = int(_test_config.get('limit', 5))
    
    @pytest.fixture(autouse=True)
    def setup(self, rotowire_rss_url):
        """Setup for integration tests using URL from config."""
        self.rss_url = rotowire_rss_url
        self.tool = RotowireRSSFeedTool(rss_url=self.rss_url)
        self.tools = self.tool.get_all_tools()
        yield
    
    @pytest.mark.integration
    def test_get_nba_news_tool_integration(self, save_test_output):
        """Test get_rotowire_nba_news tool with real RSS feed.
        
        This test actually fetches NBA news from the Rotowire RSS feed.
        Sample output saved to: tests/test_outputs/test_get_nba_news_tool_integration_get_rotowire_nba_news.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_rotowire_nba_news')
        
        # Test with limit
        limit = self.TEST_LIMIT
        
        # Call real tool
        result = get_tool.invoke({"limit": limit})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_rotowire_nba_news", result)
        
        # Verify the result
        assert "Rotowire NBA News" in result
        assert "Failed" not in result
        assert len(result) > 50  # Should have substantial content
        assert str(limit) in result or "items" in result.lower()
    

