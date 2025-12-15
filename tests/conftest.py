"""
Pytest configuration and shared fixtures for ClutchAI tests.
"""

import os
import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock, Mock
from typing import List, Optional

# Add project root to Python path for imports
# This ensures tests can import agents modules when run from IDE or different directories
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@pytest.fixture
def mock_firecrawl_v2():
    """Mock Firecrawl v2 client."""
    mock_app = MagicMock()
    mock_app.scrape = MagicMock()
    mock_app.map = MagicMock()
    mock_app.crawl = MagicMock()
    return mock_app


@pytest.fixture
def mock_firecrawl_v1():
    """Mock Firecrawl v1 client."""
    mock_app = MagicMock()
    mock_app.scrape_url = MagicMock()
    mock_app.map_url = MagicMock()
    mock_app.crawl_url = MagicMock()
    return mock_app


@pytest.fixture
def firecrawl_api_key():
    """Test Firecrawl API key."""
    return "test_firecrawl_api_key_12345"


@pytest.fixture
def sample_urls():
    """Sample URLs for testing."""
    return [
        "https://sports.yahoo.com/nba/news/",
        "https://www.nba.com/news",
    ]


@pytest.fixture
def mock_scrape_response():
    """Mock response from Firecrawl scrape."""
    return {
        'markdown': '# Test Article\n\nThis is test content.',
        'title': 'Test Article Title',
        'description': 'Test article description',
        'links': ['https://example.com/link1', 'https://example.com/link2'],
    }


@pytest.fixture
def mock_map_response():
    """Mock response from Firecrawl map."""
    return {
        'links': [
            {
                'url': 'https://example.com/article1',
                'title': 'Article 1',
                'description': 'Description 1',
            },
            {
                'url': 'https://example.com/article2',
                'title': 'Article 2',
                'description': 'Description 2',
            },
        ],
        'metadata': {'total': 2},
    }


@pytest.fixture
def mock_crawl_response():
    """Mock response from Firecrawl crawl."""
    return {
        'jobId': 'test_job_123',
        'status': 'completed',
    }


@pytest.fixture
def env_firecrawl_key(monkeypatch, firecrawl_api_key):
    """Set FIRECRAWL_API_KEY environment variable."""
    monkeypatch.setenv('FIRECRAWL_API_KEY', firecrawl_api_key)
    return firecrawl_api_key


@pytest.fixture(autouse=True)
def cleanup_env(monkeypatch):
    """Clean up environment variables after each test."""
    yield
    # Cleanup happens automatically with monkeypatch


@pytest.fixture
def test_output_dir():
    """Create and return test output directory for saving test results."""
    output_dir = Path(__file__).parent / "test_outputs"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture
def save_test_output(test_output_dir, request):
    """Fixture to save test output to a file.
    
    Usage in tests:
        result = tool.invoke(...)
        save_test_output("tool_name", result)
    
    Outputs are saved to tests/test_outputs/{test_file_name}/{test_function_name}_{tool_name}.txt
    Each test file gets its own folder for better organization.
    """
    def _save_output(tool_name: str, output: str):
        """Save output to file in test file-specific folder."""
        # Get the test file name (without .py extension)
        test_file_path = Path(request.node.fspath)
        test_file_name = test_file_path.stem  # e.g., "test_dynasty_ranking" or "test_yahoo_api_integration"
        
        # Get the test function name
        test_function_name = request.node.name
        
        # Create a folder for this test file
        test_file_folder = test_output_dir / test_file_name
        test_file_folder.mkdir(exist_ok=True)
        
        # Save file inside the test file folder with test function name and tool name
        output_file = test_file_folder / f"{test_function_name}_{tool_name}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Test File: {test_file_name}\n")
            f.write(f"Test Function: {test_function_name}\n")
            f.write(f"Tool: {tool_name}\n")
            f.write("=" * 80 + "\n\n")
            f.write(output)
            f.write("\n")
        return output_file
    
    return _save_output

