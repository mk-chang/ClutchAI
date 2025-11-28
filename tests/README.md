# ClutchAI Tests

This directory contains pytest tests for the ClutchAI package.

## Running Tests

### Run all tests
```bash
pytest
```

### Run with coverage
```bash
pytest --cov=ClutchAI --cov-report=html --cov-report=term
```

### Run specific test file
```bash
pytest tests/test_fantasy_news.py
```

### Run specific test class
```bash
pytest tests/test_fantasy_news.py::TestFantasyNewsTool
```

### Run specific test function
```bash
pytest tests/test_fantasy_news.py::TestFantasyNewsTool::test_init
```

### Run with verbose output
```bash
pytest -v
```

### Run with markers
```bash
# Run only unit tests (fast, use mocks)
pytest -m unit

# Run only integration tests (slower, use real APIs)
pytest -m integration

# Run both (exclude integration tests)
pytest -m "not integration"
```

## Test Structure

- `conftest.py`: Shared fixtures and pytest configuration
- `test_base.py`: Tests for base tool classes (`ClutchAITool`, `FirecrawlTool`)
- `test_fantasy_news.py`: **Unit tests** for `FantasyNewsTool` (uses mocks)
- `test_fantasy_news_integration.py`: **Integration tests** for `FantasyNewsTool` (uses real API)

## Unit Tests vs Integration Tests

### Unit Tests (`test_fantasy_news.py`)
- ✅ Fast execution (no network calls)
- ✅ Use mocked API responses
- ✅ Don't require API keys
- ✅ Test tool logic and formatting
- ✅ Run by default

### Integration Tests (`test_fantasy_news_integration.py`)
- 🔌 **Test real API connections**
- 🔌 **Require `FIRECRAWL_API_KEY` environment variable**
- 🔌 Make actual network calls
- 🔌 Verify API integration works end-to-end
- 🔌 Output saved to `tests/test_outputs/`

**To run integration tests:**
```bash
# Set API key
export FIRECRAWL_API_KEY=your_api_key_here

# Run integration tests
pytest -m integration tests/test_fantasy_news_integration.py -v

# Or run a specific integration test
pytest tests/test_fantasy_news_integration.py::TestFantasyNewsToolIntegration::test_scrape_url_tool_integration -v
```

**Note:** Integration tests are automatically skipped if `FIRECRAWL_API_KEY` is not set.

## Test Coverage

Current test coverage includes:

### Base Tools (`test_base.py`)
- ✅ `ClutchAITool._format_response()` with various data types
- ✅ `FirecrawlTool` initialization (v1 and v2)
- ✅ Error handling (missing API key, missing package)
- ✅ Environment variable handling
- ✅ Debug logging

### Fantasy News Tool

**Unit Tests** (`test_fantasy_news.py`):
- ✅ Tool initialization and configuration
- ✅ `get_all_tools()` method
- ✅ `scrape_url` tool (mocked)
- ✅ `map_url` tool (mocked)
- ✅ Error handling
- ✅ Output formatting

**Integration Tests** (`test_fantasy_news_integration.py`):
- 🔌 `scrape_url` tool with real Firecrawl API
- 🔌 `map_url` tool with real Firecrawl API
- 🔌 Actual API response validation

## Writing New Tests

When adding new tools, follow these patterns:

1. **Use fixtures from `conftest.py`** for common mocks
2. **Mock external dependencies** (Firecrawl, APIs, etc.)
3. **Test both success and error cases**
4. **Test parameter validation**
5. **Use descriptive test names** following `test_<what>_<condition>` pattern

### Example Test Structure

```python
class TestNewTool:
    """Tests for NewTool."""
    
    @patch.dict('os.environ', {'FIRECRAWL_API_KEY': 'test_key'})
    @patch('ClutchAI.tools.base.Firecrawl')
    def test_init(self, mock_firecrawl_class):
        """Test NewTool initialization."""
        # Test implementation
        pass
    
    def test_success_case(self):
        """Test successful operation."""
        # Test implementation
        pass
    
    def test_error_handling(self):
        """Test error handling."""
        # Test implementation
        pass
```

## Test Outputs

Test outputs are saved to `tests/test_outputs/` for easy viewing:
- Unit test outputs: Shows formatted mock responses
- Integration test outputs: Shows actual API responses

Files are automatically created when tests run and are ignored by git.

## Mocking External Dependencies

**Unit tests** mock all external dependencies to avoid:
- Network calls during tests
- API rate limits
- External service dependencies
- Slow test execution

Key mocks in unit tests:
- `Firecrawl` / `FirecrawlApp` clients
- API responses
- Environment variables

**Integration tests** make real API calls and require:
- Valid API keys (set via environment variables)
- Network connectivity
- API service availability

## Continuous Integration

Tests should be run in CI/CD pipelines before merging code. Ensure all tests pass and maintain good coverage.


