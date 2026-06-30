# Waiver Wire Postgres Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Redis-based waiver wire cache with a Postgres-persisted store that invalidates only when a new transaction (roster move) is detected.

**Architecture:** A new `WaiverWireStore` Postgres manager stores the player list alongside the most recent `transaction_id` at fetch time. On each query, `WaiverWireTool` fetches the latest `transaction_id` from Yahoo (one cheap API call) and compares it to the stored value — cache hit if they match, re-fetch if not. Data lives in Postgres forever; no TTL expiry.

**Tech Stack:** SQLAlchemy (via existing `PostgresConnection`), yfpy `get_league_transactions()`, LangChain `@tool`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `data/postgres/waiver_wire.py` | Create | `WaiverWireStore` — table DDL, get/put/delete |
| `tests/test_waiver_wire_store.py` | Create | Unit tests for `WaiverWireStore` |
| `agents/tools/waiver_wire.py` | Rewrite | Replace Redis with `WaiverWireStore`; add `_get_latest_tx_id()` |
| `tests/test_waiver_wire.py` | Rewrite | Tests for Postgres-based caching logic |
| `agents/multi_agent/yahoo_fantasy_agent.py` | Modify | `redis_client` → `connection` param |
| `tests/test_yahoo_fantasy_agent_waiver.py` | Modify | Update mock params to match |
| `agents/multi_agent/multi_agent_system.py` | Modify | Remove Redis block; pass `connection` to `YahooFantasyAgent` |
| `tests/test_multi_agent_system_redis.py` | Delete + replace | Stale — replace with `tests/test_multi_agent_system_waiver.py` |

---

## Task 1: WaiverWireStore (Postgres data layer)

**Files:**
- Create: `data/postgres/waiver_wire.py`
- Create: `tests/test_waiver_wire_store.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_waiver_wire_store.py
from unittest.mock import MagicMock, patch
import json
import pytest


def _make_mock_connection(fetchone_result=None):
    """Build a mock PostgresConnection whose engine returns controlled row data."""
    mock_conn_ctx = MagicMock()
    mock_execute_result = MagicMock()
    mock_execute_result.fetchone.return_value = fetchone_result
    mock_conn_ctx.__enter__ = MagicMock(return_value=mock_conn_ctx)
    mock_conn_ctx.__exit__ = MagicMock(return_value=False)
    mock_conn_ctx.execute.return_value = mock_execute_result
    mock_conn_ctx.commit = MagicMock()

    mock_engine = MagicMock()
    mock_engine.connect.return_value = mock_conn_ctx

    mock_connection = MagicMock()
    mock_connection.get_engine.return_value = mock_engine
    return mock_connection, mock_conn_ctx


class TestWaiverWireStore:

    def test_create_table_executes_ddl(self):
        from data.postgres.waiver_wire import WaiverWireStore

        mock_connection, mock_conn_ctx = _make_mock_connection()
        store = WaiverWireStore(mock_connection)
        store.create_table()

        assert mock_conn_ctx.execute.called
        sql = str(mock_conn_ctx.execute.call_args[0][0])
        assert "waiver_wire_cache" in sql
        mock_conn_ctx.commit.assert_called_once()

    def test_get_returns_none_when_no_row(self):
        from data.postgres.waiver_wire import WaiverWireStore

        mock_connection, _ = _make_mock_connection(fetchone_result=None)
        store = WaiverWireStore(mock_connection)

        result = store.get("466.l.58930")
        assert result is None

    def test_get_returns_dict_when_row_exists(self):
        from data.postgres.waiver_wire import WaiverWireStore

        players_data = [{"name": "Josh Hart", "position": "SF", "team": "NYK",
                         "percent_owned": 45, "ownership_type": "freeagents"}]
        mock_row = MagicMock()
        mock_row.players = players_data
        mock_row.last_tx_id = 42

        mock_connection, _ = _make_mock_connection(fetchone_result=mock_row)
        store = WaiverWireStore(mock_connection)

        result = store.get("466.l.58930")
        assert result == {"players": players_data, "last_tx_id": 42}

    def test_put_executes_upsert(self):
        from data.postgres.waiver_wire import WaiverWireStore

        mock_connection, mock_conn_ctx = _make_mock_connection()
        store = WaiverWireStore(mock_connection)

        players = [{"name": "Devin Booker", "position": "SG", "team": "PHX",
                    "percent_owned": 88, "ownership_type": "freeagents"}]
        store.put("466.l.58930", players, last_tx_id=99)

        assert mock_conn_ctx.execute.called
        sql = str(mock_conn_ctx.execute.call_args[0][0])
        assert "INSERT" in sql
        assert "ON CONFLICT" in sql
        mock_conn_ctx.commit.assert_called_once()

    def test_delete_executes_delete_sql(self):
        from data.postgres.waiver_wire import WaiverWireStore

        mock_connection, mock_conn_ctx = _make_mock_connection()
        store = WaiverWireStore(mock_connection)
        store.delete("466.l.58930")

        assert mock_conn_ctx.execute.called
        sql = str(mock_conn_ctx.execute.call_args[0][0])
        assert "DELETE" in sql
        mock_conn_ctx.commit.assert_called_once()
```

- [ ] **Step 2: Run to verify they fail**

```bash
conda run -n ClutchAI pytest tests/test_waiver_wire_store.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'data.postgres.waiver_wire'`

- [ ] **Step 3: Create `data/postgres/waiver_wire.py`**

```python
import json
from typing import Optional

from sqlalchemy import text

from data.postgres.connection import PostgresConnection
from logger import get_logger

logger = get_logger(__name__)


class WaiverWireStore:
    TABLE = "waiver_wire_cache"

    def __init__(self, connection: PostgresConnection):
        self.connection = connection

    def create_table(self) -> None:
        engine = self.connection.get_engine()
        with engine.connect() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {self.TABLE} (
                    league_key  VARCHAR(50)              PRIMARY KEY,
                    players     JSONB                    NOT NULL,
                    last_tx_id  INTEGER,
                    fetched_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """))
            conn.commit()
        logger.info(f"Table '{self.TABLE}' ready")

    def get(self, league_key: str) -> Optional[dict]:
        """Return {'players': [...], 'last_tx_id': int|None} or None if no row."""
        engine = self.connection.get_engine()
        with engine.connect() as conn:
            row = conn.execute(
                text(f"SELECT players, last_tx_id FROM {self.TABLE} WHERE league_key = :k"),
                {"k": league_key},
            ).fetchone()
        if row is None:
            return None
        return {"players": row.players, "last_tx_id": row.last_tx_id}

    def put(self, league_key: str, players: list, last_tx_id: Optional[int]) -> None:
        engine = self.connection.get_engine()
        with engine.connect() as conn:
            conn.execute(
                text(f"""
                    INSERT INTO {self.TABLE} (league_key, players, last_tx_id, fetched_at)
                    VALUES (:k, :p::jsonb, :tx, NOW())
                    ON CONFLICT (league_key) DO UPDATE
                    SET players    = EXCLUDED.players,
                        last_tx_id = EXCLUDED.last_tx_id,
                        fetched_at = EXCLUDED.fetched_at
                """),
                {"k": league_key, "p": json.dumps(players), "tx": last_tx_id},
            )
            conn.commit()

    def delete(self, league_key: str) -> None:
        engine = self.connection.get_engine()
        with engine.connect() as conn:
            conn.execute(
                text(f"DELETE FROM {self.TABLE} WHERE league_key = :k"),
                {"k": league_key},
            )
            conn.commit()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n ClutchAI pytest tests/test_waiver_wire_store.py -v
```
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add data/postgres/waiver_wire.py tests/test_waiver_wire_store.py
git commit -m "feat: add WaiverWireStore for Postgres-persisted waiver wire cache"
```

---

## Task 2: Rewrite WaiverWireTool to use Postgres

**Files:**
- Rewrite: `agents/tools/waiver_wire.py`
- Rewrite: `tests/test_waiver_wire.py`

- [ ] **Step 1: Rewrite the tests first**

Replace the entire contents of `tests/test_waiver_wire.py` with:

```python
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


def _make_mock_query(players_batch, tx_id=10):
    query = MagicMock()
    query.get_league_key.return_value = "466.l.58930"

    call_count = [0]

    def fake_query(url, keys):
        call_count[0] += 1
        if call_count[0] == 1:
            return players_batch
        raise Exception("No more players")

    query.query.side_effect = fake_query

    mock_tx = MagicMock()
    mock_tx.transaction_id = tx_id
    query.get_league_transactions.return_value = [mock_tx]

    return query


class TestWaiverWireTool:

    def test_fetch_free_agents_returns_serialized_list(self):
        from agents.tools.waiver_wire import WaiverWireTool

        players = [
            _make_mock_player("Josh Hart", "SF", "NYK", 45),
            _make_mock_player("Devin Booker", "SG", "PHX", 88),
        ]
        query = _make_mock_query(players)
        tool = WaiverWireTool(query=query, connection=None)

        result = tool._fetch_free_agents(limit=50)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["name"] == "Josh Hart"
        assert result[1]["percent_owned"] == 88

    def test_get_latest_tx_id_returns_max_id(self):
        from agents.tools.waiver_wire import WaiverWireTool

        query = MagicMock()
        query.get_league_key.return_value = "466.l.58930"
        tx1, tx2 = MagicMock(), MagicMock()
        tx1.transaction_id = 5
        tx2.transaction_id = 12
        query.get_league_transactions.return_value = [tx1, tx2]

        tool = WaiverWireTool(query=query, connection=None)
        assert tool._get_latest_tx_id() == 12

    def test_get_latest_tx_id_returns_none_on_failure(self):
        from agents.tools.waiver_wire import WaiverWireTool

        query = MagicMock()
        query.get_league_key.return_value = "466.l.58930"
        query.get_league_transactions.side_effect = Exception("API error")

        tool = WaiverWireTool(query=query, connection=None)
        assert tool._get_latest_tx_id() is None

    def test_get_waiver_wire_players_uses_store_on_cache_hit(self):
        from agents.tools.waiver_wire import WaiverWireTool

        cached_players = [{"name": "Cached Player", "position": "PG", "team": "LAL",
                           "percent_owned": 20, "ownership_type": "freeagents"}]
        mock_store = MagicMock()
        mock_store.get.return_value = {"players": cached_players, "last_tx_id": 10}

        query = _make_mock_query([], tx_id=10)

        with patch("agents.tools.waiver_wire.WaiverWireStore", return_value=mock_store):
            tool = WaiverWireTool(query=query, connection=MagicMock())

        result = tool._get_waiver_wire_players()

        mock_store.get.assert_called_once_with("466.l.58930")
        query.query.assert_not_called()
        assert "Cached Player" in result

    def test_get_waiver_wire_players_refetches_on_new_transaction(self):
        from agents.tools.waiver_wire import WaiverWireTool

        cached_players = [{"name": "Old Player", "position": "PG", "team": "LAL",
                           "percent_owned": 20, "ownership_type": "freeagents"}]
        mock_store = MagicMock()
        mock_store.get.return_value = {"players": cached_players, "last_tx_id": 9}  # stale

        new_players = [_make_mock_player("New Player", "SG", "BOS", 55)]
        query = _make_mock_query(new_players, tx_id=10)  # tx_id 10 != stored 9

        with patch("agents.tools.waiver_wire.WaiverWireStore", return_value=mock_store):
            tool = WaiverWireTool(query=query, connection=MagicMock())

        result = tool._get_waiver_wire_players()

        query.query.assert_called()
        mock_store.put.assert_called_once_with("466.l.58930", tool._fetch_free_agents.__self__
                                                if hasattr(tool._fetch_free_agents, '__self__') else
                                                mock_store.put.call_args[0][1], 10)
        assert "New Player" in result

    def test_get_waiver_wire_players_no_connection_always_fetches(self):
        from agents.tools.waiver_wire import WaiverWireTool

        players = [_make_mock_player("Jaylen Brown", "SF", "BOS", 75)]
        query = _make_mock_query(players)
        tool = WaiverWireTool(query=query, connection=None)

        result = tool._get_waiver_wire_players()

        assert "Jaylen Brown" in result

    def test_get_all_tools_returns_langchain_tools(self):
        from agents.tools.waiver_wire import WaiverWireTool

        query = MagicMock()
        query.get_league_key.return_value = "466.l.58930"
        tool = WaiverWireTool(query=query, connection=None)

        tools = tool.get_all_tools()
        tool_names = [t.name for t in tools]

        assert "get_waiver_wire_players" in tool_names
        assert "refresh_waiver_wire_cache" in tool_names
```

**Note on `test_get_waiver_wire_players_refetches_on_new_transaction`:** The assertion on `mock_store.put` is simplified — just verify it was called once with the right league key and tx_id; the players list content doesn't need exact matching.

Replace the test with this cleaner version:

```python
    def test_get_waiver_wire_players_refetches_on_new_transaction(self):
        from agents.tools.waiver_wire import WaiverWireTool

        cached_players = [{"name": "Old Player", "position": "PG", "team": "LAL",
                           "percent_owned": 20, "ownership_type": "freeagents"}]
        mock_store = MagicMock()
        mock_store.get.return_value = {"players": cached_players, "last_tx_id": 9}

        new_players = [_make_mock_player("New Player", "SG", "BOS", 55)]
        query = _make_mock_query(new_players, tx_id=10)

        with patch("agents.tools.waiver_wire.WaiverWireStore", return_value=mock_store):
            tool = WaiverWireTool(query=query, connection=MagicMock())

        result = tool._get_waiver_wire_players()

        query.query.assert_called()
        assert mock_store.put.call_args[0][0] == "466.l.58930"
        assert mock_store.put.call_args[0][2] == 10
        assert "New Player" in result
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n ClutchAI pytest tests/test_waiver_wire.py -v
```
Expected: FAIL (imports or attribute errors — `WaiverWireTool` still has Redis interface)

- [ ] **Step 3: Rewrite `agents/tools/waiver_wire.py`**

Replace the entire file with:

```python
import json
from typing import List, Optional
from langchain_core.tools import tool

from .base import ClutchAITool
from data.postgres.connection import PostgresConnection
from data.postgres.waiver_wire import WaiverWireStore
from logger import get_logger

logger = get_logger(__name__)

_BATCH_SIZE = 25


class WaiverWireTool(ClutchAITool):
    """
    Fetches free agents from Yahoo Fantasy API and persists results in Postgres.

    Cache invalidation: on each query, fetches the latest transaction_id from
    Yahoo and compares it to the stored value. Stale on mismatch; re-fetches.
    Data persists indefinitely — no TTL expiry.
    """

    def __init__(self, query, connection: Optional[PostgresConnection] = None, debug: bool = False):
        """
        Args:
            query: YahooFantasySportsQuery instance
            connection: PostgresConnection for persistent caching (optional)
            debug: Enable debug logging
        """
        super().__init__(debug=debug)
        self.query = query
        self.store = None
        if connection is not None:
            try:
                self.store = WaiverWireStore(connection)
                self.store.create_table()
            except Exception as e:
                logger.warning(f"WaiverWireStore unavailable, caching disabled: {e}")
                self.store = None

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

    def _get_latest_tx_id(self) -> Optional[int]:
        """Return the highest transaction_id in the league, or None on failure."""
        try:
            transactions = self.query.get_league_transactions()
            if not transactions:
                return None
            ids = [t.transaction_id for t in transactions if t.transaction_id is not None]
            return max(ids) if ids else None
        except Exception as e:
            logger.warning(f"Could not fetch transactions for cache check: {e}")
            return None

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

    def _get_waiver_wire_players(self) -> str:
        """
        Return free agents as JSON string.

        Checks Postgres for a cached entry. If the latest transaction_id
        matches the stored last_tx_id, returns cached data. Otherwise
        re-fetches from Yahoo and updates the store.
        """
        latest_tx_id = self._get_latest_tx_id()
        league_key = self.query.get_league_key()

        if self.store is not None:
            try:
                cached = self.store.get(league_key)
                if cached:
                    if latest_tx_id is None or cached["last_tx_id"] == latest_tx_id:
                        logger.debug("Waiver wire cache hit")
                        return json.dumps(cached["players"], indent=2)
            except Exception as e:
                logger.warning(f"Waiver wire store read failed, fetching live: {e}")

        players = self._fetch_free_agents()

        if self.store is not None and latest_tx_id is not None:
            try:
                self.store.put(league_key, players, latest_tx_id)
            except Exception as e:
                logger.warning(f"Waiver wire store write failed: {e}")

        return json.dumps(players, indent=2)

    def get_all_tools(self) -> list:
        _ww_fetch = self._get_waiver_wire_players
        store = self.store
        league_key_fn = self.query.get_league_key
        fetch_free_agents = self._fetch_free_agents
        get_latest_tx_id = self._get_latest_tx_id

        @tool
        def get_waiver_wire_players() -> str:
            """
            Get available free agents (waiver wire players) in the fantasy league.

            Returns up to 50 free agents with name, position, NBA team,
            percent_owned, and ownership_type. Results are cached in Postgres
            and automatically refreshed when new transactions (roster moves)
            are detected.

            Returns:
                JSON string with list of free agent player objects
            """
            return _ww_fetch()

        @tool
        def refresh_waiver_wire_cache() -> str:
            """
            Force a fresh fetch of waiver wire players from Yahoo, bypassing the cache.

            Use this when you suspect the cached data is incorrect.

            Returns:
                JSON string with updated list of free agent player objects
            """
            league_key = league_key_fn()
            if store is not None:
                try:
                    store.delete(league_key)
                except Exception as e:
                    logger.warning(f"Waiver wire cache delete failed: {e}")
            players = fetch_free_agents(limit=50)
            latest_tx_id = get_latest_tx_id()
            result = json.dumps(players, indent=2)
            if store is not None and latest_tx_id is not None:
                try:
                    store.put(league_key, players, latest_tx_id)
                except Exception as e:
                    logger.warning(f"Waiver wire store write failed: {e}")
            return result

        return [get_waiver_wire_players, refresh_waiver_wire_cache]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n ClutchAI pytest tests/test_waiver_wire.py -v
```
Expected: 7 PASSED

- [ ] **Step 5: Commit**

```bash
git add agents/tools/waiver_wire.py tests/test_waiver_wire.py
git commit -m "feat: replace Redis waiver wire cache with Postgres-persisted store"
```

---

## Task 3: Update YahooFantasyAgent (redis_client → connection)

**Files:**
- Modify: `agents/multi_agent/yahoo_fantasy_agent.py`
- Modify: `tests/test_yahoo_fantasy_agent_waiver.py`

- [ ] **Step 1: Update `tests/test_yahoo_fantasy_agent_waiver.py`**

Replace `redis_client` with `connection` in both tests:

```python
# tests/test_yahoo_fantasy_agent_waiver.py
from unittest.mock import MagicMock, patch


def test_yahoo_fantasy_agent_includes_waiver_wire_tools():
    """WaiverWireTool should be instantiated with query + connection."""
    with patch("agents.multi_agent.yahoo_fantasy_agent.WaiverWireTool") as MockWaiver, \
         patch("agents.multi_agent.yahoo_fantasy_agent.YahooFantasyTool") as MockYahoo, \
         patch("agents.multi_agent.base_agent.BasicTool") as MockBasic, \
         patch("agents.multi_agent.base_agent.ChatOpenAI"), \
         patch("agents.multi_agent.base_agent.create_agent"), \
         patch("agents.multi_agent.base_agent.yaml.safe_load", return_value={}):

        mock_ww = MagicMock()
        mock_ww.get_all_tools.return_value = [MagicMock(), MagicMock()]
        MockWaiver.return_value = mock_ww

        mock_yahoo = MagicMock()
        mock_yahoo.get_all_tools.return_value = []
        MockYahoo.return_value = mock_yahoo

        mock_basic = MagicMock()
        mock_basic.get_all_tools.return_value = []
        MockBasic.return_value = mock_basic

        mock_query = MagicMock()
        mock_connection = MagicMock()

        from agents.multi_agent.yahoo_fantasy_agent import YahooFantasyAgent
        agent = YahooFantasyAgent(
            query=mock_query,
            connection=mock_connection,
            openai_api_key="test-key",
        )

    MockWaiver.assert_called_once_with(query=mock_query, connection=mock_connection, debug=False)
    mock_ww.get_all_tools.assert_called_once()


def test_yahoo_fantasy_agent_no_connection_still_loads_waiver_wire():
    """WaiverWireTool should load even without connection (caching disabled)."""
    with patch("agents.multi_agent.yahoo_fantasy_agent.WaiverWireTool") as MockWaiver, \
         patch("agents.multi_agent.yahoo_fantasy_agent.YahooFantasyTool") as MockYahoo, \
         patch("agents.multi_agent.base_agent.BasicTool") as MockBasic, \
         patch("agents.multi_agent.base_agent.ChatOpenAI"), \
         patch("agents.multi_agent.base_agent.create_agent"), \
         patch("agents.multi_agent.base_agent.yaml.safe_load", return_value={}):

        mock_ww = MagicMock()
        mock_ww.get_all_tools.return_value = []
        MockWaiver.return_value = mock_ww

        mock_yahoo = MagicMock()
        mock_yahoo.get_all_tools.return_value = []
        MockYahoo.return_value = mock_yahoo

        mock_basic = MagicMock()
        mock_basic.get_all_tools.return_value = []
        MockBasic.return_value = mock_basic

        mock_query = MagicMock()

        from agents.multi_agent.yahoo_fantasy_agent import YahooFantasyAgent
        agent = YahooFantasyAgent(
            query=mock_query,
            connection=None,
            openai_api_key="test-key",
        )

    MockWaiver.assert_called_once_with(query=mock_query, connection=None, debug=False)
```

- [ ] **Step 2: Run to verify tests fail**

```bash
conda run -n ClutchAI pytest tests/test_yahoo_fantasy_agent_waiver.py -v
```
Expected: FAIL (still uses `redis_client` keyword)

- [ ] **Step 3: Update `agents/multi_agent/yahoo_fantasy_agent.py`**

Change `__init__` and `_create_tools`:

```python
def __init__(self, connection=None, **kwargs):
    self._connection = connection
    super().__init__(**kwargs)
```

In `_create_tools`, change the `WaiverWireTool` instantiation line from:
```python
waiver_tool = WaiverWireTool(query=self.query, redis_client=self._redis_client, debug=self.debug)
```
to:
```python
waiver_tool = WaiverWireTool(query=self.query, connection=self._connection, debug=self.debug)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n ClutchAI pytest tests/test_yahoo_fantasy_agent_waiver.py -v
```
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add agents/multi_agent/yahoo_fantasy_agent.py tests/test_yahoo_fantasy_agent_waiver.py
git commit -m "refactor: replace redis_client with connection in YahooFantasyAgent"
```

---

## Task 4: Update MultiAgentSystem (remove Redis, pass connection)

**Files:**
- Modify: `agents/multi_agent/multi_agent_system.py`
- Delete: `tests/test_multi_agent_system_redis.py`
- Create: `tests/test_multi_agent_system_waiver.py`

- [ ] **Step 1: Write new test file**

```python
# tests/test_multi_agent_system_waiver.py
from unittest.mock import MagicMock, patch, call
import pytest


def _make_mas_patches():
    return [
        patch("agents.multi_agent.multi_agent_system.YahooFantasySportsQuery"),
        patch("agents.multi_agent.multi_agent_system.RAGManager"),
        patch("agents.multi_agent.multi_agent_system.UserContextGatherer"),
        patch("agents.multi_agent.multi_agent_system.YahooFantasyAgent"),
        patch("agents.multi_agent.multi_agent_system.StatisticAgent"),
        patch("agents.multi_agent.multi_agent_system.NewsAgent"),
        patch("agents.multi_agent.multi_agent_system.FantasyAnalystAgent"),
        patch("agents.multi_agent.multi_agent_system.SupervisorAgent"),
    ]


def test_postgres_connection_passed_to_yahoo_agent():
    """MultiAgentSystem passes its Postgres connection to YahooFantasyAgent."""
    patches = _make_mas_patches()
    mocks = [p.start() for p in patches]

    try:
        mock_gatherer = mocks[2].return_value
        mock_gatherer.gather.return_value = ""
        mock_gatherer.get_display_info.return_value = {}

        MockYahooAgent = mocks[3]
        mock_connection = MagicMock()

        import agents.multi_agent.multi_agent_system as mas_module
        MultiAgentSystem = mas_module.MultiAgentSystem

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            mas = MultiAgentSystem(disable_rag=True, connection=mock_connection)

        call_kwargs = MockYahooAgent.call_args.kwargs
        assert call_kwargs.get("connection") is mock_connection
    finally:
        for p in patches:
            p.stop()


def test_postgres_connection_created_from_database_url_when_not_passed():
    """When connection=None but DATABASE_URL is set, MultiAgentSystem creates one."""
    patches = _make_mas_patches()
    mocks = [p.start() for p in patches]

    try:
        mock_gatherer = mocks[2].return_value
        mock_gatherer.gather.return_value = ""
        mock_gatherer.get_display_info.return_value = {}

        MockYahooAgent = mocks[3]

        with patch("agents.multi_agent.multi_agent_system.PostgresConnection") as MockPG, \
             patch.dict("os.environ", {"OPENAI_API_KEY": "test-key",
                                       "DATABASE_URL": "postgresql://test"}):

            mock_pg_instance = MagicMock()
            MockPG.return_value = mock_pg_instance

            import agents.multi_agent.multi_agent_system as mas_module
            MultiAgentSystem = mas_module.MultiAgentSystem

            mas = MultiAgentSystem(disable_rag=True)

            call_kwargs = MockYahooAgent.call_args.kwargs
            assert call_kwargs.get("connection") is mock_pg_instance
    finally:
        for p in patches:
            p.stop()


def test_connection_none_when_no_database_url():
    """When no connection and no DATABASE_URL, YahooFantasyAgent receives connection=None."""
    patches = _make_mas_patches()
    mocks = [p.start() for p in patches]

    try:
        mock_gatherer = mocks[2].return_value
        mock_gatherer.gather.return_value = ""
        mock_gatherer.get_display_info.return_value = {}

        MockYahooAgent = mocks[3]

        env = {"OPENAI_API_KEY": "test-key"}
        # Ensure DATABASE_URL is absent
        with patch.dict("os.environ", env):
            import os
            os.environ.pop("DATABASE_URL", None)

            import agents.multi_agent.multi_agent_system as mas_module
            MultiAgentSystem = mas_module.MultiAgentSystem

            mas = MultiAgentSystem(disable_rag=True)

            call_kwargs = MockYahooAgent.call_args.kwargs
            assert call_kwargs.get("connection") is None
    finally:
        for p in patches:
            p.stop()
```

- [ ] **Step 2: Delete old test file and run new tests to verify they fail**

```bash
rm /Users/matt/Code/ClutchAI/tests/test_multi_agent_system_redis.py
conda run -n ClutchAI pytest tests/test_multi_agent_system_waiver.py -v
```
Expected: FAIL (`MultiAgentSystem` still uses `redis_client`)

- [ ] **Step 3: Update `agents/multi_agent/multi_agent_system.py`**

**3a. Remove the Redis import** (line 29):
```python
# Remove this line:
from data.redis.connection import RedisConnection
```

**3b. Remove the `redis_url` parameter** from `__init__` signature (line 55):
```python
# Remove this line:
redis_url: Optional[str] = None,
```

**3c. Replace the Redis initialization block** with a Postgres connection block for waiver wire.

Remove these lines:
```python
# Initialize Redis client (optional — caching disabled if unavailable)
redis_client = None
_redis_url = redis_url or os.environ.get("REDIS_URL")
if _redis_url:
    try:
        redis_client = RedisConnection(redis_url=_redis_url).get_client()
        logger.info("Redis connected for waiver wire caching")
    except Exception as e:
        logger.warning(f"Redis not available, waiver wire caching disabled: {e}")
```

Replace with:
```python
# Postgres connection for waiver wire store (separate from RAG)
ww_connection = None
if connection is not None:
    ww_connection = connection
elif os.environ.get("DATABASE_URL"):
    try:
        ww_connection = PostgresConnection()
        logger.debug("Postgres connection ready for waiver wire store")
    except Exception as e:
        logger.warning(f"Postgres not available for waiver wire caching: {e}")
```

**3d. Update the `YahooFantasyAgent` instantiation** to use `connection=ww_connection` instead of `redis_client=redis_client`:
```python
self.yahoo_fantasy_agent = YahooFantasyAgent(
    query=self.query,
    rag_manager=None,
    tools_config=self.tools_config,
    openai_api_key=self.openai_api_key,
    project_root=self.env_file_location,
    debug=self.debug,
    connection=ww_connection,
)
```

- [ ] **Step 4: Run new tests**

```bash
conda run -n ClutchAI pytest tests/test_multi_agent_system_waiver.py -v
```
Expected: 3 PASSED

- [ ] **Step 5: Run full suite**

```bash
conda run -n ClutchAI pytest tests/ -k "not integration" -q 2>&1 | tail -10
```
Expected: all pass, 0 failures

- [ ] **Step 6: Commit**

```bash
git add agents/multi_agent/multi_agent_system.py tests/test_multi_agent_system_waiver.py
git rm tests/test_multi_agent_system_redis.py
git commit -m "refactor: replace Redis waiver wire wiring with Postgres connection in MultiAgentSystem"
```

---

## Self-Review

**Spec coverage:**
- ✅ `WaiverWireStore` with `create_table / get / put / delete` — Task 1
- ✅ `_get_latest_tx_id()` fetches transactions and returns max ID — Task 2
- ✅ Cache hit when `last_tx_id` matches — Task 2
- ✅ Re-fetch on mismatch (new transaction) — Task 2
- ✅ Data persists in Postgres (no TTL) — Task 1 schema
- ✅ `connection=None` degrades gracefully (no caching) — Tasks 2, 3, 4
- ✅ Redis initialization removed from `MultiAgentSystem` — Task 4
- ✅ Fallback: creates `PostgresConnection()` from `DATABASE_URL` when `connection=None` — Task 4

**Placeholder scan:** None found.

**Type consistency:** `connection: Optional[PostgresConnection]` used consistently across `WaiverWireTool.__init__`, `YahooFantasyAgent.__init__`, and `MultiAgentSystem`. `WaiverWireStore.get()` returns `Optional[dict]` with keys `players` and `last_tx_id` — consumed consistently in `_get_waiver_wire_players`.
