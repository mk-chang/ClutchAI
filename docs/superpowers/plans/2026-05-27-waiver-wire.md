# Waiver Wire Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a waiver wire tool that fetches free agents from Yahoo Fantasy API and caches results in Redis to avoid expensive repeated API calls.

**Architecture:** A new `WaiverWireTool` queries the Yahoo Fantasy API directly for `status=FA` players (including ownership/percent_owned metadata) and stores the JSON-serialized result in Redis with a 1-hour TTL, keyed by league. A new `RedisConnection` class in `data/redis/connection.py` mirrors the existing `PostgresConnection` pattern. The tool is added to `YahooFantasyAgent`'s toolset; `MultiAgentSystem` wires Redis in during initialization.

**Tech Stack:** `redis` (Python client), yfpy's internal `query()` method for custom Yahoo API URLs, LangChain `@tool` decorator, Railway Redis plugin (`REDIS_URL` env var).

---

## File Map

| File | Status | Purpose |
|------|--------|---------|
| `data/redis/connection.py` | Create | RedisConnection wrapping `redis.Redis.from_url()` |
| `data/redis/__init__.py` | Create | Empty init |
| `agents/tools/waiver_wire.py` | Create | WaiverWireTool with Redis caching |
| `agents/multi_agent/yahoo_fantasy_agent.py` | Modify | Accept `redis_client`, add WaiverWireTool |
| `agents/multi_agent/multi_agent_system.py` | Modify | Create RedisConnection, pass redis_client |
| `config/multiagent_config.yaml` | Modify | Update yahoo_fantasy system prompt |
| `requirements.txt` | Modify | Add `redis` |
| `tests/test_waiver_wire.py` | Create | Unit tests with mocked Redis and yfpy |

---

## Task 1: Redis Connection

**Files:**
- Create: `data/redis/__init__.py`
- Create: `data/redis/connection.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_redis_connection.py
from unittest.mock import patch, MagicMock
import pytest


def test_redis_connection_from_env():
    with patch("redis.Redis.from_url") as mock_from_url:
        mock_client = MagicMock()
        mock_from_url.return_value = mock_client

        from data.redis.connection import RedisConnection
        conn = RedisConnection(redis_url="redis://localhost:6379")

        mock_from_url.assert_called_once_with("redis://localhost:6379", decode_responses=True)
        assert conn.get_client() is mock_client


def test_redis_connection_from_env_var():
    with patch.dict("os.environ", {"REDIS_URL": "redis://env-host:6379"}):
        with patch("redis.Redis.from_url") as mock_from_url:
            mock_from_url.return_value = MagicMock()

            from data.redis.connection import RedisConnection
            import importlib
            import data.redis.connection
            importlib.reload(data.redis.connection)
            conn = data.redis.connection.RedisConnection()

            mock_from_url.assert_called_once_with("redis://env-host:6379", decode_responses=True)


def test_redis_connection_raises_without_url():
    with patch.dict("os.environ", {}, clear=True):
        from data.redis.connection import RedisConnection
        with pytest.raises(ValueError, match="REDIS_URL"):
            RedisConnection()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/matt/Code/ClutchAI
conda run -n ClutchAI pytest tests/test_redis_connection.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'data.redis'`

- [ ] **Step 3: Add `redis` to requirements and install**

In `requirements.txt`, after the `# Railway deployment` section, add:

```
# Redis caching
redis
```

Then install:
```bash
conda run -n ClutchAI pip install redis
```

- [ ] **Step 4: Create `data/redis/__init__.py`**

```python
```
(empty file)

- [ ] **Step 5: Create `data/redis/connection.py`**

```python
import os
import redis


class RedisConnection:

    def __init__(self, redis_url: str = None):
        url = redis_url or os.environ.get("REDIS_URL")
        if not url:
            raise ValueError(
                "REDIS_URL environment variable is required. "
                "Railway injects this automatically when a Redis plugin is attached."
            )
        self._client = redis.Redis.from_url(url, decode_responses=True)

    def get_client(self) -> redis.Redis:
        return self._client
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
conda run -n ClutchAI pytest tests/test_redis_connection.py -v
```
Expected: PASS (3 tests)

- [ ] **Step 7: Commit**

```bash
git add data/redis/__init__.py data/redis/connection.py tests/test_redis_connection.py requirements.txt
git commit -m "feat: add RedisConnection class for Railway Redis plugin"
```

---

## Task 2: WaiverWireTool

**Files:**
- Create: `agents/tools/waiver_wire.py`
- Create: `tests/test_waiver_wire.py`

### Background: How the Yahoo API call works

`YahooFantasySportsQuery.query()` is the internal method that all public methods (like `get_league_players`) use. It accepts a URL and a list of keys to drill down into the response.

Free agents URL pattern:
```
https://fantasysports.yahooapis.com/fantasy/v2/league/{league_key}/players;status=FA;start={start};count=25;out=ownership,percent_owned
```

`status=FA` filters to free agents only. `out=ownership,percent_owned` adds roster status and % owned.
The `query()` call uses `["league", "players"]` as the data_key_list to extract the players list.

The response objects are yfpy `Player` instances. Each has:
- `player.name.full` — player full name
- `player.primary_position` — e.g. "PG", "SF"  
- `player.editorial_team_abbr` — NBA team abbreviation
- `player.percent_owned` — yfpy PercentOwned object with `.value` (int, 0-100)
- `player.ownership` — yfpy Ownership object with `.ownership_type` ("freeagents", "waivers", "team")

### Cache design

- Key: `clutchai:waiver_wire:{league_key}` (e.g. `clutchai:waiver_wire:466.l.58930`)
- Value: JSON string of serialized player list
- TTL: 3600 seconds (1 hour)
- Redis client is optional — if `None`, the tool skips caching and always fetches live

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_waiver_wire.py
import json
from unittest.mock import MagicMock, patch
import pytest


def _make_mock_player(name, position, team, percent_owned, ownership_type="freeagents"):
    player = MagicMock()
    player.name.full = name
    player.primary_position = position
    player.editorial_team_abbr = team
    player.percent_owned.value = percent_owned
    player.ownership.ownership_type = ownership_type
    return player


def _make_mock_query(players_batch):
    """Return a mock YahooFantasySportsQuery that returns players_batch then raises."""
    query = MagicMock()
    query.get_league_key.return_value = "466.l.58930"

    call_count = [0]

    def fake_query(url, keys):
        call_count[0] += 1
        if call_count[0] == 1:
            return players_batch
        raise Exception("No more players")  # triggers pagination stop

    query.query.side_effect = fake_query
    return query


class TestWaiverWireTool:

    def test_fetch_free_agents_returns_serialized_list(self):
        from agents.tools.waiver_wire import WaiverWireTool

        players = [
            _make_mock_player("Nikola Jokic", "C", "DEN", 99, "team"),
            _make_mock_player("Josh Hart", "SF", "NYK", 45, "freeagents"),
        ]
        query = _make_mock_query(players)
        tool = WaiverWireTool(query=query, redis_client=None)

        result = tool._fetch_free_agents(limit=25)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["name"] == "Nikola Jokic"
        assert result[1]["percent_owned"] == 45

    def test_get_waiver_wire_players_uses_redis_cache(self):
        from agents.tools.waiver_wire import WaiverWireTool

        cached_data = json.dumps([{"name": "Cached Player", "position": "PG", "team": "LAL", "percent_owned": 20, "ownership_type": "freeagents"}])
        redis_client = MagicMock()
        redis_client.get.return_value = cached_data

        query = MagicMock()
        query.get_league_key.return_value = "466.l.58930"
        tool = WaiverWireTool(query=query, redis_client=redis_client)

        result = tool._get_waiver_wire_players(limit=50)

        redis_client.get.assert_called_once_with("clutchai:waiver_wire:466.l.58930")
        query.query.assert_not_called()
        assert "Cached Player" in result

    def test_get_waiver_wire_players_populates_cache_on_miss(self):
        from agents.tools.waiver_wire import WaiverWireTool

        redis_client = MagicMock()
        redis_client.get.return_value = None  # cache miss

        players = [_make_mock_player("Devin Booker", "SG", "PHX", 88, "freeagents")]
        query = _make_mock_query(players)
        tool = WaiverWireTool(query=query, redis_client=redis_client)

        result = tool._get_waiver_wire_players(limit=50)

        redis_client.setex.assert_called_once()
        call_args = redis_client.setex.call_args
        assert call_args[0][0] == "clutchai:waiver_wire:466.l.58930"
        assert call_args[0][1] == 3600
        assert "Devin Booker" in result

    def test_get_waiver_wire_players_no_redis_always_fetches(self):
        from agents.tools.waiver_wire import WaiverWireTool

        players = [_make_mock_player("Jaylen Brown", "SF", "BOS", 75, "freeagents")]
        query = _make_mock_query(players)
        tool = WaiverWireTool(query=query, redis_client=None)

        result = tool._get_waiver_wire_players(limit=50)

        assert "Jaylen Brown" in result

    def test_get_all_tools_returns_langchain_tools(self):
        from agents.tools.waiver_wire import WaiverWireTool

        query = MagicMock()
        query.get_league_key.return_value = "466.l.58930"
        tool = WaiverWireTool(query=query, redis_client=None)

        tools = tool.get_all_tools()
        tool_names = [t.name for t in tools]

        assert "get_waiver_wire_players" in tool_names
        assert "refresh_waiver_wire_cache" in tool_names
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n ClutchAI pytest tests/test_waiver_wire.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'agents.tools.waiver_wire'`

- [ ] **Step 3: Implement `agents/tools/waiver_wire.py`**

```python
import json
from typing import List, Optional
from langchain_core.tools import tool

from .base import ClutchAITool
from logger import get_logger

logger = get_logger(__name__)

_CACHE_TTL = 3600  # 1 hour
_BATCH_SIZE = 25


class WaiverWireTool(ClutchAITool):
    """
    Fetches free agents from Yahoo Fantasy API and caches results in Redis.

    Cache key: clutchai:waiver_wire:{league_key}
    TTL: 3600 seconds (1 hour)

    Uses yfpy's query() method directly to add status=FA and ownership data
    to the standard league players URL.
    """

    def __init__(self, query, redis_client=None, debug: bool = False):
        """
        Args:
            query: YahooFantasySportsQuery instance
            redis_client: redis.Redis client (optional; disables caching if None)
            debug: Enable debug logging
        """
        super().__init__(debug=debug)
        self.query = query
        self.redis_client = redis_client

    def _cache_key(self) -> str:
        return f"clutchai:waiver_wire:{self.query.get_league_key()}"

    def _serialize_player(self, player) -> dict:
        try:
            return {
                "name": player.name.full,
                "position": player.primary_position,
                "team": player.editorial_team_abbr,
                "percent_owned": player.percent_owned.value,
                "ownership_type": player.ownership.ownership_type,
            }
        except Exception as e:
            logger.debug(f"Error serializing player: {e}")
            return {"name": str(player), "position": "", "team": "", "percent_owned": 0, "ownership_type": ""}

    def _fetch_free_agents(self, limit: int = 50) -> List[dict]:
        """Fetch free agents from Yahoo API with pagination."""
        league_key = self.query.get_league_key()
        players = []
        start = 0

        while len(players) < limit:
            count = min(_BATCH_SIZE, limit - len(players))
            url = (
                f"https://fantasysports.yahooapis.com/fantasy/v2/league/{league_key}/players;"
                f"status=FA;start={start};count={count};out=ownership,percent_owned"
            )
            try:
                batch = self.query.query(url, ["league", "players"])
                if not batch:
                    break
                if not isinstance(batch, list):
                    batch = [batch]
                players.extend(self._serialize_player(p) for p in batch)
                if len(batch) < count:
                    break
                start += count
            except Exception as e:
                logger.debug(f"No more free agents at start={start}: {e}")
                break

        logger.info(f"Fetched {len(players)} free agents from Yahoo API")
        return players

    def _get_waiver_wire_players(self, limit: int = 50) -> str:
        """Return free agents as JSON string, using Redis cache when available."""
        if self.redis_client is not None:
            cached = self.redis_client.get(self._cache_key())
            if cached:
                logger.debug("Waiver wire cache hit")
                return cached

        players = self._fetch_free_agents(limit=limit)
        result = json.dumps(players, indent=2)

        if self.redis_client is not None:
            self.redis_client.setex(self._cache_key(), _CACHE_TTL, result)
            logger.debug(f"Cached {len(players)} free agents (TTL={_CACHE_TTL}s)")

        return result

    def get_all_tools(self) -> list:
        get_waiver_wire_players = self._get_waiver_wire_players
        cache_key = self._cache_key
        redis_client = self.redis_client
        fetch_free_agents = self._fetch_free_agents

        @tool
        def get_waiver_wire_players(limit: int = 50) -> str:
            """
            Get available free agents (waiver wire players) in the fantasy league.

            Returns a JSON list of free agents with name, position, NBA team,
            percent_owned, and ownership_type. Results are cached for 1 hour.

            Args:
                limit: Maximum number of players to return (default 50)

            Returns:
                JSON string with list of free agent player objects
            """
            return get_waiver_wire_players(limit=limit)

        @tool
        def refresh_waiver_wire_cache() -> str:
            """
            Force a fresh fetch of waiver wire players from Yahoo, bypassing the Redis cache.

            Use this when you need up-to-date data after recent roster moves.

            Returns:
                JSON string with updated list of free agent player objects
            """
            if redis_client is not None:
                redis_client.delete(cache_key())
                logger.info("Waiver wire cache cleared")
            players = fetch_free_agents(limit=50)
            result = json.dumps(players, indent=2)
            if redis_client is not None:
                redis_client.setex(cache_key(), _CACHE_TTL, result)
            return result

        return [get_waiver_wire_players, refresh_waiver_wire_cache]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n ClutchAI pytest tests/test_waiver_wire.py -v
```
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add agents/tools/waiver_wire.py tests/test_waiver_wire.py
git commit -m "feat: add WaiverWireTool with Redis caching"
```

---

## Task 3: Wire WaiverWireTool into YahooFantasyAgent

**Files:**
- Modify: `agents/multi_agent/yahoo_fantasy_agent.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_yahoo_fantasy_agent_waiver.py
from unittest.mock import MagicMock, patch


def test_yahoo_fantasy_agent_includes_waiver_wire_tools():
    """WaiverWireTool tools should appear in the agent's tool list when redis_client provided."""
    with patch("agents.multi_agent.yahoo_fantasy_agent.WaiverWireTool") as MockWaiver:
        mock_ww_instance = MagicMock()
        mock_ww_instance.get_all_tools.return_value = [MagicMock(name="get_waiver_wire_players"), MagicMock(name="refresh_waiver_wire_cache")]
        MockWaiver.return_value = mock_ww_instance

        mock_query = MagicMock()
        mock_redis = MagicMock()

        with patch("agents.multi_agent.yahoo_fantasy_agent.YahooFantasyTool") as MockYahoo:
            mock_yahoo_instance = MagicMock()
            mock_yahoo_instance.get_all_tools.return_value = []
            MockYahoo.return_value = mock_yahoo_instance

            with patch("agents.multi_agent.yahoo_fantasy_agent.BasicTool") as MockBasic:
                mock_basic_instance = MagicMock()
                mock_basic_instance.get_all_tools.return_value = []
                MockBasic.return_value = mock_basic_instance

                with patch("agents.multi_agent.base_agent.ChatOpenAI"), \
                     patch("agents.multi_agent.base_agent.create_agent"), \
                     patch("agents.multi_agent.base_agent.yaml.safe_load", return_value={}):
                    from agents.multi_agent.yahoo_fantasy_agent import YahooFantasyAgent
                    agent = YahooFantasyAgent(
                        query=mock_query,
                        redis_client=mock_redis,
                        openai_api_key="test-key",
                    )

        MockWaiver.assert_called_once_with(query=mock_query, redis_client=mock_redis)
        mock_ww_instance.get_all_tools.assert_called_once()


def test_yahoo_fantasy_agent_skips_waiver_wire_without_redis():
    """WaiverWireTool should still be added even without redis_client (caching is optional)."""
    with patch("agents.multi_agent.yahoo_fantasy_agent.WaiverWireTool") as MockWaiver:
        mock_ww_instance = MagicMock()
        mock_ww_instance.get_all_tools.return_value = []
        MockWaiver.return_value = mock_ww_instance

        mock_query = MagicMock()

        with patch("agents.multi_agent.yahoo_fantasy_agent.YahooFantasyTool") as MockYahoo, \
             patch("agents.multi_agent.yahoo_fantasy_agent.BasicTool") as MockBasic, \
             patch("agents.multi_agent.base_agent.ChatOpenAI"), \
             patch("agents.multi_agent.base_agent.create_agent"), \
             patch("agents.multi_agent.base_agent.yaml.safe_load", return_value={}):
            mock_yahoo_instance = MagicMock()
            mock_yahoo_instance.get_all_tools.return_value = []
            MockYahoo.return_value = mock_yahoo_instance
            mock_basic_instance = MagicMock()
            mock_basic_instance.get_all_tools.return_value = []
            MockBasic.return_value = mock_basic_instance

            from agents.multi_agent.yahoo_fantasy_agent import YahooFantasyAgent
            agent = YahooFantasyAgent(
                query=mock_query,
                redis_client=None,
                openai_api_key="test-key",
            )

        MockWaiver.assert_called_once_with(query=mock_query, redis_client=None)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n ClutchAI pytest tests/test_yahoo_fantasy_agent_waiver.py -v
```
Expected: FAIL — `YahooFantasyAgent.__init__` has no `redis_client` parameter

- [ ] **Step 3: Modify `agents/multi_agent/yahoo_fantasy_agent.py`**

Current file head:
```python
from yfpy.query import YahooFantasySportsQuery

from agents.multi_agent.base_agent import BaseAgent
from agents.tools.yahoo_api import YahooFantasyTool
```

Change to:
```python
from typing import Optional

from yfpy.query import YahooFantasySportsQuery

from agents.multi_agent.base_agent import BaseAgent
from agents.tools.yahoo_api import YahooFantasyTool
from agents.tools.waiver_wire import WaiverWireTool
```

The `YahooFantasyAgent` class currently has no `__init__`. Add one (the base class `__init__` handles everything via `_create_tools`). We need to pass `redis_client` through.

Replace the class body so `redis_client` can be stored before `_create_tools` is called by `super().__init__`:

```python
class YahooFantasyAgent(BaseAgent):
    """
    Yahoo Fantasy Agent that specializes in Yahoo Fantasy API data.
    """

    def __init__(self, redis_client=None, **kwargs):
        self._redis_client = redis_client
        super().__init__(**kwargs)

    def _get_config_section(self) -> str:
        return 'yahoo_fantasy'

    def _get_default_system_prompt(self) -> str:
        return """You are a Yahoo Fantasy specialist agent for fantasy basketball analysis.
Your role is to gather comprehensive data from the Yahoo Fantasy API:
- League data: standings, settings, teams, players
- Team data: rosters, stats, matchups, draft results
- Player data: stats, ownership, draft analysis
- Transaction data: trades, waiver claims, adds/drops
- Waiver wire: available free agents with ownership percentages

When given a research task, use all relevant Yahoo Fantasy tools to gather data.
Be thorough and provide structured data that can be easily analyzed."""

    def _create_tools(self) -> List:
        tools = list(super()._create_base_tools())

        if self.query is None:
            self.logger.warning("YahooFantasySportsQuery not provided. Yahoo Fantasy tools will not be available.")
        else:
            try:
                yahoo_tool = YahooFantasyTool(query=self.query, debug=self.debug)
                tools.extend(yahoo_tool.get_all_tools())
                self.logger.debug("Yahoo Fantasy tools loaded")
            except Exception as e:
                self.logger.warning(f"Yahoo Fantasy tools not available: {e}")

            try:
                waiver_tool = WaiverWireTool(query=self.query, redis_client=self._redis_client, debug=self.debug)
                tools.extend(waiver_tool.get_all_tools())
                self.logger.debug("Waiver wire tools loaded")
            except Exception as e:
                self.logger.warning(f"Waiver wire tools not available: {e}")

        self.logger.info(f"Yahoo Fantasy Agent initialized with {len(tools)} tools")
        return tools
```

Also add `List` to the imports at the top of the file:
```python
from typing import List, Optional
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n ClutchAI pytest tests/test_yahoo_fantasy_agent_waiver.py -v
```
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add agents/multi_agent/yahoo_fantasy_agent.py tests/test_yahoo_fantasy_agent_waiver.py
git commit -m "feat: wire WaiverWireTool into YahooFantasyAgent"
```

---

## Task 4: Wire Redis into MultiAgentSystem

**Files:**
- Modify: `agents/multi_agent/multi_agent_system.py`

- [ ] **Step 1: Locate the two spots to change**

Open [agents/multi_agent/multi_agent_system.py](agents/multi_agent/multi_agent_system.py).

Spot A — imports (around line 10–25): add RedisConnection import.

Spot B — `__init__` signature and body (around line 35–60): add `redis_url` param and create `RedisConnection`.

Spot C — `YahooFantasyAgent` instantiation (around line 130): pass `redis_client`.

- [ ] **Step 2: Add import**

In the imports section, after `from data.postgres.connection import PostgresConnection`, add:

```python
from data.redis.connection import RedisConnection
```

- [ ] **Step 3: Add `redis_url` param to `__init__`**

In the `__init__` signature, after `disable_rag: Optional[bool] = None,`, add:

```python
redis_url: Optional[str] = None,
```

In the docstring args section, add:
```
redis_url: Redis URL for waiver wire caching (or from REDIS_URL env var). Optional — caching disabled if not set.
```

- [ ] **Step 4: Initialize Redis connection in `__init__` body**

After the `self.disable_rag = disable_rag` line, add:

```python
# Initialize Redis client for waiver wire caching (optional)
self.redis_client = None
try:
    redis_conn = RedisConnection(redis_url=redis_url)
    self.redis_client = redis_conn.get_client()
    logger.info("Redis connected for waiver wire caching")
except ValueError:
    logger.info("REDIS_URL not set — waiver wire caching disabled")
except Exception as e:
    logger.warning(f"Redis connection failed — waiver wire caching disabled: {e}")
```

- [ ] **Step 5: Pass `redis_client` to `YahooFantasyAgent`**

Find the `YahooFantasyAgent(...)` instantiation block and add `redis_client=self.redis_client`:

```python
self.yahoo_fantasy_agent = YahooFantasyAgent(
    query=self.query,
    rag_manager=None,
    tools_config=self.tools_config,
    openai_api_key=self.openai_api_key,
    project_root=self.env_file_location,
    debug=self.debug,
    redis_client=self.redis_client,
)
```

- [ ] **Step 6: Run existing tests to check nothing is broken**

```bash
conda run -n ClutchAI pytest tests/ -v --ignore=tests/test_redis_connection.py -k "not integration" 2>&1 | tail -30
```
Expected: All previously passing tests still pass

- [ ] **Step 7: Commit**

```bash
git add agents/multi_agent/multi_agent_system.py
git commit -m "feat: wire Redis into MultiAgentSystem for waiver wire caching"
```

---

## Task 5: Update Supervisor System Prompt

**Files:**
- Modify: `config/multiagent_config.yaml`

- [ ] **Step 1: Update yahoo_fantasy system prompt**

In `config/multiagent_config.yaml`, find the `yahoo_fantasy:` section. The current `system_prompt` lists:
```
- Transaction data: trades, waiver claims, adds/drops
```

Add one line after it:
```
- Waiver wire: available free agents sorted by % owned, with refresh capability
```

Also add to the `supervisor:` system prompt under `WHEN TO USE SPECIALIZED RESEARCH AGENTS` → `Yahoo Fantasy Agent` bullet:

```
- Yahoo Fantasy Agent: Use for Yahoo league data, rosters, standings, matchups, player ownership, transactions, and waiver wire free agents
```

- [ ] **Step 2: Commit**

```bash
git add config/multiagent_config.yaml
git commit -m "docs: update system prompts to mention waiver wire tool"
```

---

## Task 6: Integration Smoke Test

- [ ] **Step 1: Run the full test suite**

```bash
conda run -n ClutchAI pytest tests/ -v -k "not integration" 2>&1 | tail -40
```
Expected: All unit tests pass

- [ ] **Step 2: Manual smoke test — tool with no Redis**

```python
# Run in a Python shell inside the ClutchAI conda env
# conda run -n ClutchAI python
from unittest.mock import MagicMock, patch

mock_players = [MagicMock()]
mock_players[0].name.full = "Josh Hart"
mock_players[0].primary_position = "SF"
mock_players[0].editorial_team_abbr = "NYK"
mock_players[0].percent_owned.value = 45
mock_players[0].ownership.ownership_type = "freeagents"

query = MagicMock()
query.get_league_key.return_value = "466.l.58930"
query.query.side_effect = [mock_players, Exception("done")]

from agents.tools.waiver_wire import WaiverWireTool
tool = WaiverWireTool(query=query, redis_client=None)
result = tool._get_waiver_wire_players(limit=10)
print(result)
# Expected: JSON with Josh Hart
```

- [ ] **Step 3: Commit final state if any cleanup needed**

```bash
git add -p  # review any remaining changes
git commit -m "chore: finalize waiver wire feature"
```

---

## Self-Review

**Spec coverage:**
- ✅ Waiver wire free agents fetched from Yahoo Fantasy API
- ✅ Results cached in Redis with TTL
- ✅ Cache miss → fetch → write Redis
- ✅ Cache hit → return from Redis (no Yahoo API call)
- ✅ Redis optional (works without it, caching disabled)
- ✅ Tool exposed to YahooFantasyAgent via LangChain `@tool`
- ✅ `refresh_waiver_wire_cache` tool for force-refresh
- ✅ `RedisConnection` follows existing `PostgresConnection` pattern

**Placeholder scan:** None found.

**Type consistency:**
- `WaiverWireTool._get_waiver_wire_players()` returns `str` (JSON)
- `WaiverWireTool._fetch_free_agents()` returns `List[dict]`
- `WaiverWireTool.get_all_tools()` returns `list` (LangChain tools)
- `YahooFantasyAgent.__init__` stores `redis_client` as `self._redis_client` before `super().__init__()` calls `_create_tools()`
- `RedisConnection.get_client()` returns `redis.Redis` instance
