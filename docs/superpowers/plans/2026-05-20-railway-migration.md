# Railway Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace GCP infrastructure (Cloud Run + Cloud SQL + Secret Manager) with Railway, keeping Streamlit UI and all agent code untouched.

**Architecture:** Three Railway services share one project — web app, PostgreSQL plugin (with pgvector), and cron job. The only code changes are in `data/postgres/connection.py` (Cloud SQL Connector → standard psycopg2 via `DATABASE_URL`), `data/postgres/schema.py` (env var rename), `requirements.txt`, and `Dockerfile`.

**Tech Stack:** Railway, PostgreSQL + pgvector, SQLAlchemy + psycopg2, existing Streamlit app

---

### Task 1: Rewrite `data/postgres/connection.py`

Remove the Google Cloud SQL Python Connector. Replace with a standard SQLAlchemy engine built from `DATABASE_URL`. Keep `get_engine()` and the context manager interface — nothing else in the codebase touches the internals.

**Files:**
- Modify: `data/postgres/connection.py`
- Create: `tests/test_postgres_connection.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_postgres_connection.py`:

```python
import pytest
from data.postgres.connection import PostgresConnection


def test_raises_without_database_url(monkeypatch):
    monkeypatch.delenv('DATABASE_URL', raising=False)
    with pytest.raises(ValueError, match="DATABASE_URL"):
        PostgresConnection()


def test_creates_engine_from_database_url(monkeypatch):
    monkeypatch.setenv('DATABASE_URL', 'postgresql://user:pass@localhost:5432/testdb')
    conn = PostgresConnection()
    assert conn.get_engine() is not None
    conn.close()


def test_accepts_explicit_database_url(monkeypatch):
    monkeypatch.delenv('DATABASE_URL', raising=False)
    conn = PostgresConnection(database_url='postgresql://user:pass@localhost:5432/testdb')
    assert conn.get_engine() is not None
    conn.close()


def test_context_manager(monkeypatch):
    monkeypatch.setenv('DATABASE_URL', 'postgresql://user:pass@localhost:5432/testdb')
    with PostgresConnection() as conn:
        assert conn.get_engine() is not None
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_postgres_connection.py -v
```

Expected: 4 failures — `ImportError` or `TypeError` because the current `PostgresConnection.__init__` doesn't accept `database_url`.

- [ ] **Step 3: Rewrite `data/postgres/connection.py`**

Replace the entire file contents with:

```python
import os
from sqlalchemy import create_engine, Engine


class PostgresConnection:
    """
    PostgreSQL connection using a standard DATABASE_URL.
    Provides a SQLAlchemy engine compatible with langchain-postgres PGVector.
    """

    def __init__(self, database_url: str = None):
        url = database_url or os.environ.get('DATABASE_URL')
        if not url:
            raise ValueError(
                "DATABASE_URL environment variable is required. "
                "Railway injects this automatically when a PostgreSQL plugin is attached."
            )
        # Railway provides postgresql:// URLs; SQLAlchemy needs the driver specified.
        if url.startswith('postgresql://'):
            url = url.replace('postgresql://', 'postgresql+psycopg2://', 1)
        self._engine = create_engine(url, pool_pre_ping=True, pool_recycle=3600)

    def get_engine(self) -> Engine:
        return self._engine

    def close(self):
        if hasattr(self, '_engine') and self._engine:
            self._engine.dispose()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_postgres_connection.py -v
```

Expected: 4 passing. The engine is created lazily — no real database needed for these tests.

- [ ] **Step 5: Commit**

```bash
git add data/postgres/connection.py tests/test_postgres_connection.py
git commit -m "Replace Cloud SQL Connector with standard psycopg2 connection via DATABASE_URL"
```

---

### Task 2: Update env var names in `data/postgres/schema.py`

`CLOUDSQL_VECTOR_TABLE` and `CLOUDSQL_APP_TABLE` are GCP-prefixed names that should just be `VECTOR_TABLE` and `APP_TABLE` post-migration.

**Files:**
- Modify: `data/postgres/schema.py`

- [ ] **Step 1: Update `get_default_table_name()`**

In `data/postgres/schema.py`, find `get_default_table_name()` and replace the body:

```python
def get_default_table_name() -> str:
    table_name = os.environ.get('VECTOR_TABLE')
    if table_name:
        return table_name
    raise ValueError(
        "Vector table name is required. "
        "Set VECTOR_TABLE environment variable to specify the table name."
    )
```

- [ ] **Step 2: Update `get_app_table_name()`**

Find `get_app_table_name()` in `data/postgres/schema.py`. Replace every reference to `CLOUDSQL_APP_TABLE` with `APP_TABLE` in that function's body and its docstring.

- [ ] **Step 3: Update test skip condition in `tests/test_vectordb_connection.py`**

The integration test currently skips if GCP env vars are absent. Replace the `pytestmark` skip condition at the top of the file:

```python
pytestmark = pytest.mark.skipif(
    not os.environ.get('DATABASE_URL'),
    reason="DATABASE_URL not set. Integration tests require a live PostgreSQL connection."
)
```

Also update the `setup` fixture — the current `PostgresConnection()` call will now work since it reads `DATABASE_URL` directly. No other changes needed in the test body.

- [ ] **Step 4: Run unit tests to confirm nothing broke**

```bash
pytest tests/test_postgres_connection.py -v
```

Expected: 4 passing. The integration tests in `test_vectordb_connection.py` will skip (no `DATABASE_URL` locally).

- [ ] **Step 5: Commit**

```bash
git add data/postgres/schema.py tests/test_vectordb_connection.py
git commit -m "Rename CLOUDSQL_VECTOR_TABLE -> VECTOR_TABLE, update integration test skip condition"
```

---

### Task 3: Update `requirements.txt`

Remove the GCP-specific connector package. `psycopg2-binary` is already present.

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Remove the Cloud SQL connector line**

In `requirements.txt`, delete this line:

```
cloud-sql-python-connector[pg8000,psycopg2]
```

The comment above the vector DB block can be updated from:

```
# Vector Database - PostgreSQL with pgvector (Google Cloud SQL)
```

to:

```
# Vector Database - PostgreSQL with pgvector
```

- [ ] **Step 2: Verify the package list looks correct**

```bash
grep -i "psycopg\|cloud-sql\|pg8000\|langchain-postgres" requirements.txt
```

Expected output:
```
langchain-postgres
psycopg2-binary
```

No `cloud-sql-python-connector` or `pg8000` lines.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "Remove cloud-sql-python-connector from requirements"
```

---

### Task 4: Update `Dockerfile`

Railway injects `PORT` dynamically. The hardcoded `8080` won't work.

**Files:**
- Modify: `Dockerfile`

- [ ] **Step 1: Update the CMD line**

Replace the last line of `Dockerfile`:

```dockerfile
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8080"]
```

with:

```dockerfile
CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port ${PORT:-8080}"]
```

The `${PORT:-8080}` fallback lets you still run the container locally without setting `PORT`.

- [ ] **Step 2: Verify the Docker build succeeds locally**

```bash
docker build -t clutchai-test .
```

Expected: build completes without errors.

- [ ] **Step 3: Commit**

```bash
git add Dockerfile
git commit -m "Use Railway PORT env var in Dockerfile CMD"
```

---

### Task 5: Railway project setup (manual steps)

These are one-time setup steps in the Railway dashboard and CLI. No code changes.

**Prerequisites:** Railway account at railway.app, Railway CLI installed (`npm install -g @railway/cli`).

- [ ] **Step 1: Create Railway project**

```bash
railway login
railway init
```

Name the project `clutchai`. Choose "Empty project" when prompted.

- [ ] **Step 2: Add PostgreSQL plugin**

In the Railway dashboard: open the project → **+ New** → **Database** → **PostgreSQL**. Railway creates the database and sets `DATABASE_URL` automatically for all services in the project.

- [ ] **Step 3: Enable pgvector extension**

Click the PostgreSQL service → **Connect** → open the psql console (or use any postgres client with the connection string from the Railway dashboard). Run:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Verify with:

```sql
SELECT * FROM pg_extension WHERE extname = 'vector';
```

Expected: one row returned.

- [ ] **Step 4: Set environment variables on the web service**

In Railway: open the web service → **Variables**. Add each of these (values from your current GCP `.env` or Secret Manager):

| Variable | Value |
|----------|-------|
| `OPENAI_API_KEY` | your key |
| `YAHOO_CLIENT_ID` | your key |
| `YAHOO_CLIENT_SECRET` | your key |
| `YAHOO_ACCESS_TOKEN_JSON` | the full JSON blob from `build_yahoo_token_json.py` |
| `YAHOO_LEAGUE_ID` | `58930` |
| `YAHOO_REDIRECT_URI` | your Railway app URL (shown after first deploy, or custom domain) |
| `RUNTIME_ENVIRONMENT` | `docker` |
| `VECTOR_TABLE` | `vectorstore` |
| `DISABLE_RAG` | `false` |

`DATABASE_URL` is already injected automatically — do not set it manually.

- [ ] **Step 5: Connect GitHub repo and deploy**

In Railway: web service → **Settings** → **Source** → connect GitHub repo, select the `optimization` branch (or `main` after merge). Railway will auto-detect the Dockerfile and deploy.

Watch the build logs. Expected: `Streamlit` starts and the service URL appears in the dashboard.

---

### Task 6: Populate vectorstore on Railway PostgreSQL

Re-run the existing pipeline scripts against the new database. The scripts read `DATABASE_URL` from the environment.

**Files:** No code changes — run existing scripts.

- [ ] **Step 1: Get the Railway DATABASE_URL**

```bash
railway variables --service clutchai-db
```

Copy the `DATABASE_URL` value. It will look like `postgresql://postgres:password@hostname:port/railway`.

- [ ] **Step 2: Set DATABASE_URL and VECTOR_TABLE locally, then run the base knowledge pipeline**

```bash
export DATABASE_URL="<paste Railway DATABASE_URL here>"
export VECTOR_TABLE="vectorstore"
python scripts/pipelines/update_base_knowledge.py
```

Expected: script completes and prints ingestion counts.

- [ ] **Step 3: Run the LockedOn podcast pipeline**

```bash
python scripts/pipelines/update_lockedon_knowledge.py
```

Expected: script completes and prints ingestion counts.

- [ ] **Step 4: Run the full vectorstore update**

```bash
python scripts/pipelines/update_vector_database.py
```

Expected: script completes without errors.

- [ ] **Step 5: Verify data was loaded**

Connect to the Railway PostgreSQL instance via the psql console in the Railway dashboard and run:

```sql
SELECT COUNT(*) FROM langchain_pg_embedding;
```

Expected: row count greater than 0 (should match your previous Cloud SQL counts).

---

### Task 7: Set up Railway cron job

Create a cron job service that runs the vectorstore update scripts on a weekly schedule.

**Files:** No code changes.

- [ ] **Step 1: Create the cron job service in Railway**

In Railway dashboard: **+ New** → **Cron Job**. Connect the same GitHub repo and Dockerfile.

Set the start command to:

```bash
sh -c "python scripts/pipelines/update_base_knowledge.py && python scripts/pipelines/update_lockedon_knowledge.py && python scripts/pipelines/update_vector_database.py"
```

Set the cron schedule (e.g., weekly on Sunday at 3am UTC):

```
0 3 * * 0
```

- [ ] **Step 2: Add environment variables to the cron service**

The cron service needs the same variables as the web service. In Railway: cron service → **Variables** → copy all variables from the web service (Railway has a "Copy from service" option).

`DATABASE_URL` is injected automatically since the PostgreSQL plugin is in the same project.

- [ ] **Step 3: Trigger a manual run to verify**

In Railway dashboard: cron service → **Trigger** → run now. Watch the logs.

Expected: all three update scripts complete without errors.

---

### Task 8: Verify the full deployment

End-to-end check that the deployed app works correctly.

- [ ] **Step 1: Open the deployed app**

Go to the Railway web service URL (shown in the dashboard). The Streamlit app should load.

- [ ] **Step 2: Verify agent initialization**

The spinner "Initializing Multi-Agent System..." should appear and resolve within ~30 seconds. If it hangs or errors, check Railway logs for the web service.

- [ ] **Step 3: Test a RAG-backed query**

Ask the app a question that would require the knowledge base, e.g.:

> "What are the latest fantasy basketball trends you know about?"

Expected: the agent responds with content sourced from the vectorstore, not just a generic answer.

- [ ] **Step 4: Test a Yahoo API query**

Ask:

> "Summarize my team's performance this week."

Expected: agent fetches live Yahoo Fantasy data and responds. If Yahoo OAuth fails, check `YAHOO_ACCESS_TOKEN_JSON` in the Railway env vars — it may need to be regenerated locally via `scripts/pipelines/build_yahoo_token_json.py`.

- [ ] **Step 5: (Optional) Clean up GCP resources**

Once Railway is confirmed working, shut down GCP to stop billing:

```bash
gcloud run services delete clutchai --region us-central1
```

The Cloud SQL instance and Secret Manager secrets can be deleted from the GCP console. Keep the GCP project around until you're sure nothing else depends on it.
