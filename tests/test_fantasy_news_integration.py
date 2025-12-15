"""
Integration tests for FantasyNewsTool that test actual API connections.

These tests require:
- FIRECRAWL_API_KEY environment variable set
- Real Firecrawl API access

Run with: pytest -m integration tests/test_fantasy_news_integration.py
"""

import os
import yaml
from pathlib import Path
import pytest

from agents.tools.fantasy_news import FantasyNewsTool


def _load_test_config():
    """Load test configuration from test_config.yaml file."""
    config_path = Path(__file__).parent / 'test_config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        return config.get('fantasy_news', {})
    return {}


# Skip all integration tests if FIRECRAWL_API_KEY is not set
pytestmark = pytest.mark.skipif(
    not os.environ.get('FIRECRAWL_API_KEY'),
    reason="FIRECRAWL_API_KEY environment variable not set. Integration tests require real API key."
)


@pytest.mark.integration
class TestFantasyNewsToolIntegration:
    """Integration tests for FantasyNewsTool with real API calls."""
    
    # Load test configuration from YAML file
    _test_config = _load_test_config()
    
    # Test URL for scraping
    # Configured in tests/test_config.yaml
    TEST_SCRAPE_URL = _test_config.get('scrape_url', 'https://www.rotowire.com/basketball/advice/')
    
    # Test URLs for mapping
    # Configured in tests/test_config.yaml
    TEST_MAP_URLS = _test_config.get('map_urls', [
        'https://sports.yahoo.com/nba/news/',
        'https://www.nba.com/news'
    ])
    
    # Test search term
    # Configured in tests/test_config.yaml
    TEST_SEARCH_TERM = _test_config.get('search_term', 'Kevin Durant')
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for integration tests."""
        self.url = "https://sports.yahoo.com/nba/news/"
        self.tool = FantasyNewsTool()
        self.tools = self.tool.get_all_tools()
        yield
    
    @pytest.mark.integration
    def test_scrape_url_tool_integration(self, save_test_output):
        """Test scrape_url tool with real Firecrawl v2 API.
        
        This test actually calls the Firecrawl v2 API to scrape content.
        Sample output saved to: tests/test_outputs/test_scrape_url_tool_integration_scrape_url.txt
        """
        scrape_tool = next(t for t in self.tools if t.name == 'scrape_url')
        
        # Test with RotoWire URL
        test_url = self.TEST_SCRAPE_URL
        
        # Call real Firecrawl v2 API
        result = scrape_tool.invoke({"url": test_url})
        
        # Save output to file for easy viewing
        output_file = save_test_output("scrape_url", result)
        
        # Verify the result
        assert f"Scraped content from {test_url}" in result
        assert "Failed" not in result  # Should not fail
        # Real API should return actual content
        assert len(result) > 100  # Should have substantial content
    
    @pytest.mark.integration
    def test_map_url_tool_integration(self, save_test_output):
        """Test map_url tool with real Firecrawl v2 API using search term 'Kevin Durant'.
        
        This test actually calls the Firecrawl v2 API to map URLs filtered by search term.
        Tests both https://sports.yahoo.com/nba/news/ and https://www.nba.com/news
        Sample output saved to: tests/test_outputs/test_map_url_tool_integration_map_url.txt
        """
        map_tool = next(t for t in self.tools if t.name == 'map_url')
        
        # URLs to test
        test_urls = self.TEST_MAP_URLS
        
        # Call real API with search term and limit to 3 links for each URL
        search_term = self.TEST_SEARCH_TERM
        all_results = []
        
        for url in test_urls:
            result = map_tool.invoke({"url": url, "search": search_term, "limit": 3})
            all_results.append(f"Results for {url}:\n{result}\n\n")
            
            # Verify the result for each URL
            assert f"Mapped URLs from {url}" in result
            assert "Failed" not in result  # Should not fail
            # Should contain links or metadata filtered by search term
            assert len(result) > 50  # Should have some content
            # The result should include the search term in the output
            assert search_term.lower() in result.lower()
        
        # Save combined output to file for easy viewing
        combined_result = "\n".join(all_results)
        output_file = save_test_output("map_url", combined_result)
    
