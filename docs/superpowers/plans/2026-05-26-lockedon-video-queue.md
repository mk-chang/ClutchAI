# LockedOn Video Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the full-channel-scan pipeline with a persistent Postgres queue so the daily cron only processes pending videos, stops immediately when both YouTube and Supadata fail, and automatically enqueues new videos published each day.

**Architecture:** A `VideoQueueManager` class owns all queue table operations. A custom `AllTranscriptSourcesExhausted` exception signals the pipeline to stop when both transcript sources fail. The cron script rewrites to: (1) check YouTube API for yesterday's new videos → upsert to queue, (2) pull pending rows from the queue, (3) process up to `max_videos_added`, stopping the run on `AllTranscriptSourcesExhausted`.

**Tech Stack:** SQLAlchemy (raw `text()` queries), Railway Postgres, `youtube-transcript-api` via `YoutubeLoader`, Supadata REST API

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `data/postgres/video_queue.py` | `VideoQueueManager` class + `AllTranscriptSourcesExhausted` exception |
| Create | `tests/test_video_queue.py` | Unit tests for `VideoQueueManager` |
| Create | `scripts/pipelines/seed_lockedon_queue.py` | One-time script: create table + seed all 655 videos |
| Modify | `data/postgres/vector_managers/youtube.py` | Raise `AllTranscriptSourcesExhausted` when both sources fail |
| Modify | `scripts/pipelines/update_lockedon_knowledge.py` | Rewrite to use queue instead of full channel scan |

---

## Task 1: VideoQueueManager + AllTranscriptSourcesExhausted

**Files:**
- Create: `data/postgres/video_queue.py`
- Create: `tests/test_video_queue.py`

### Schema

```sql
CREATE TABLE IF NOT EXISTS lockedon_video_queue (
    video_id     VARCHAR(20) PRIMARY KEY,
    title        TEXT,
    url          TEXT NOT NULL,
    publish_date DATE,
    status       VARCHAR(10) DEFAULT 'pending'
                     CHECK (status IN ('pending', 'done', 'failed')),
    attempts     INTEGER DEFAULT 0,
    last_attempted_at TIMESTAMP WITH TIME ZONE,
    added_at     TIMESTAMP WITH TIME ZONE DEFAULT NOW()
)
```

- [ ] **Step 1: Write failing tests**

Create `tests/test_video_queue.py`:

```python
"""Tests for VideoQueueManager."""
import pytest
from unittest.mock import MagicMock, patch
from data.postgres.video_queue import VideoQueueManager, AllTranscriptSourcesExhausted


@pytest.fixture
def mock_connection():
    mock_conn = MagicMock()
    mock_engine = MagicMock()
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=mock_conn)
    ctx.__exit__ = MagicMock(return_value=False)
    mock_engine.connect.return_value = ctx
    mock_pg = MagicMock()
    mock_pg.get_engine.return_value = mock_engine
    return mock_pg, mock_conn


def test_upsert_video_returns_true_when_inserted(mock_connection):
    mock_pg, mock_conn = mock_connection
    mock_conn.execute.return_value.rowcount = 1
    manager = VideoQueueManager(mock_pg)
    result = manager.upsert_video("abc123", "Title", "https://www.youtube.com/watch?v=abc123", "2025-10-01")
    assert result is True


def test_upsert_video_returns_false_on_conflict(mock_connection):
    mock_pg, mock_conn = mock_connection
    mock_conn.execute.return_value.rowcount = 0
    manager = VideoQueueManager(mock_pg)
    result = manager.upsert_video("abc123", "Title", "https://www.youtube.com/watch?v=abc123", "2025-10-01")
    assert result is False


def test_upsert_video_commits(mock_connection):
    mock_pg, mock_conn = mock_connection
    mock_conn.execute.return_value.rowcount = 1
    manager = VideoQueueManager(mock_pg)
    manager.upsert_video("abc123", "Title", "https://www.youtube.com/watch?v=abc123", "2025-10-01")
    mock_conn.commit.assert_called_once()


def test_get_pending_returns_list_of_dicts(mock_connection):
    mock_pg, mock_conn = mock_connection
    mock_row = MagicMock()
    mock_row._mapping = {
        "video_id": "abc123",
        "title": "Episode 1",
        "url": "https://www.youtube.com/watch?v=abc123",
        "publish_date": None,
    }
    mock_conn.execute.return_value.fetchall.return_value = [mock_row]
    manager = VideoQueueManager(mock_pg)
    result = manager.get_pending(limit=5)
    assert len(result) == 1
    assert result[0]["video_id"] == "abc123"
    assert result[0]["title"] == "Episode 1"


def test_get_pending_empty_queue(mock_connection):
    mock_pg, mock_conn = mock_connection
    mock_conn.execute.return_value.fetchall.return_value = []
    manager = VideoQueueManager(mock_pg)
    assert manager.get_pending() == []


def test_mark_done_executes_and_commits(mock_connection):
    mock_pg, mock_conn = mock_connection
    manager = VideoQueueManager(mock_pg)
    manager.mark_done("abc123")
    mock_conn.execute.assert_called_once()
    mock_conn.commit.assert_called_once()


def test_mark_failed_executes_and_commits(mock_connection):
    mock_pg, mock_conn = mock_connection
    manager = VideoQueueManager(mock_pg)
    manager.mark_failed("abc123")
    mock_conn.execute.assert_called_once()
    mock_conn.commit.assert_called_once()


def test_get_stats_returns_dict(mock_connection):
    mock_pg, mock_conn = mock_connection
    mock_conn.execute.return_value.fetchall.return_value = [
        ("pending", 10),
        ("done", 5),
        ("failed", 2),
    ]
    manager = VideoQueueManager(mock_pg)
    stats = manager.get_stats()
    assert stats == {"pending": 10, "done": 5, "failed": 2}


def test_all_transcript_sources_exhausted_is_exception():
    with pytest.raises(AllTranscriptSourcesExhausted):
        raise AllTranscriptSourcesExhausted("Both YouTube and Supadata failed")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/matt/Code/ClutchAI
pytest tests/test_video_queue.py -v
```

Expected: `ModuleNotFoundError: No module named 'data.postgres.video_queue'`

- [ ] **Step 3: Implement VideoQueueManager**

Create `data/postgres/video_queue.py`:

```python
from __future__ import annotations

from typing import Dict, List, Optional

from sqlalchemy import text

from data.postgres.connection import PostgresConnection
from logger import get_logger

logger = get_logger(__name__)


class AllTranscriptSourcesExhausted(Exception):
    """Both YoutubeLoader and Supadata failed — the run should stop."""


class VideoQueueManager:
    TABLE = "lockedon_video_queue"

    def __init__(self, connection: PostgresConnection):
        self.connection = connection

    def create_table(self) -> None:
        engine = self.connection.get_engine()
        with engine.connect() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {self.TABLE} (
                    video_id     VARCHAR(20) PRIMARY KEY,
                    title        TEXT,
                    url          TEXT NOT NULL,
                    publish_date DATE,
                    status       VARCHAR(10) DEFAULT 'pending'
                                     CHECK (status IN ('pending', 'done', 'failed')),
                    attempts     INTEGER DEFAULT 0,
                    last_attempted_at TIMESTAMP WITH TIME ZONE,
                    added_at     TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """))
            conn.commit()
        logger.info(f"Table '{self.TABLE}' ready")

    def upsert_video(
        self,
        video_id: str,
        title: str,
        url: str,
        publish_date: Optional[str] = None,
        status: str = "pending",
    ) -> bool:
        """Insert video if not already in queue. Returns True if inserted."""
        engine = self.connection.get_engine()
        with engine.connect() as conn:
            result = conn.execute(
                text(f"""
                    INSERT INTO {self.TABLE} (video_id, title, url, publish_date, status)
                    VALUES (:video_id, :title, :url, :publish_date, :status)
                    ON CONFLICT (video_id) DO NOTHING
                """),
                {
                    "video_id": video_id,
                    "title": title,
                    "url": url,
                    "publish_date": publish_date,
                    "status": status,
                },
            )
            conn.commit()
            return result.rowcount > 0

    def get_pending(self, limit: int = 15) -> List[Dict]:
        """Return up to `limit` pending videos, newest-first."""
        engine = self.connection.get_engine()
        with engine.connect() as conn:
            result = conn.execute(
                text(f"""
                    SELECT video_id, title, url, publish_date
                    FROM {self.TABLE}
                    WHERE status = 'pending'
                    ORDER BY publish_date DESC NULLS LAST
                    LIMIT :limit
                """),
                {"limit": limit},
            )
            return [dict(row._mapping) for row in result.fetchall()]

    def mark_done(self, video_id: str) -> None:
        engine = self.connection.get_engine()
        with engine.connect() as conn:
            conn.execute(
                text(f"""
                    UPDATE {self.TABLE}
                    SET status = 'done', last_attempted_at = NOW()
                    WHERE video_id = :video_id
                """),
                {"video_id": video_id},
            )
            conn.commit()

    def mark_failed(self, video_id: str) -> None:
        engine = self.connection.get_engine()
        with engine.connect() as conn:
            conn.execute(
                text(f"""
                    UPDATE {self.TABLE}
                    SET status = 'failed',
                        attempts = attempts + 1,
                        last_attempted_at = NOW()
                    WHERE video_id = :video_id
                """),
                {"video_id": video_id},
            )
            conn.commit()

    def get_stats(self) -> Dict[str, int]:
        engine = self.connection.get_engine()
        with engine.connect() as conn:
            result = conn.execute(
                text(f"SELECT status, COUNT(*) FROM {self.TABLE} GROUP BY status")
            )
            return {row[0]: row[1] for row in result.fetchall()}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_video_queue.py -v
```

Expected: 9 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add data/postgres/video_queue.py tests/test_video_queue.py
git commit -m "feat: add VideoQueueManager and AllTranscriptSourcesExhausted"
```

---

## Task 2: Raise AllTranscriptSourcesExhausted in youtube.py

**Files:**
- Modify: `data/postgres/vector_managers/youtube.py`

The current behavior when both YouTube and Supadata fail: logs the error, leaves `docs = None`, then raises a generic `ValueError`. The pipeline catches this, marks the video failed, and **continues**. We need it to stop the run instead.

- [ ] **Step 1: Write failing test**

Add to `tests/test_youtube_vector_manager.py`:

```python
from data.postgres.video_queue import AllTranscriptSourcesExhausted

def test_raises_all_sources_exhausted_when_supadata_also_fails():
    """When YouTube is blocked AND Supadata fails, raise AllTranscriptSourcesExhausted."""
    connection = MagicMock()
    embeddings = MagicMock()
    manager = YoutubeVectorManager(connection=connection, embeddings=MagicMock())
    manager._youtube_blocked = True  # circuit breaker already open

    with patch("data.postgres.vector_managers.youtube.YoutubeLoader") as mock_loader, \
         patch.object(manager, "_fetch_transcript_supadata", side_effect=Exception("429 Too Many Requests")), \
         patch.object(manager, "_add_documents_to_vectorstore"):
        with pytest.raises(AllTranscriptSourcesExhausted):
            manager.load_resource_content(
                url="https://www.youtube.com/watch?v=abc123",
                resource_id="abc123",
            )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_youtube_vector_manager.py::test_raises_all_sources_exhausted_when_supadata_also_fails -v
```

Expected: FAILED — `ValueError` raised instead of `AllTranscriptSourcesExhausted`

- [ ] **Step 3: Update youtube.py**

At the top of `data/postgres/vector_managers/youtube.py`, add the import after the existing imports:

```python
from data.postgres.video_queue import AllTranscriptSourcesExhausted
```

In `load_resource_content`, find the Supadata fallback block (around line 347):

```python
            try:
                docs = self._fetch_transcript_supadata(video_id_for_fallback, chunk_size)
                logger.info(f"  ✓ Supadata fallback succeeded ({len(docs)} chunks)")
            except Exception as supadata_err:
                logger.error(f"  ✗ Supadata fallback failed: {supadata_err}")
```

Replace the `except` body with:

```python
            except Exception as supadata_err:
                logger.error(f"  ✗ Supadata fallback failed: {supadata_err}")
                raise AllTranscriptSourcesExhausted(
                    f"Both YouTube and Supadata failed for {url}"
                ) from supadata_err
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_youtube_vector_manager.py -v
```

Expected: all tests PASSED (including the new one)

- [ ] **Step 5: Commit**

```bash
git add data/postgres/vector_managers/youtube.py tests/test_youtube_vector_manager.py
git commit -m "feat: raise AllTranscriptSourcesExhausted when both YouTube and Supadata fail"
```

---

## Task 3: Seed the Queue

**Files:**
- Create: `scripts/pipelines/seed_lockedon_queue.py`

One-time script. Fetches all 655 channel videos from YouTube API, seeds the queue table, and marks already-ingested videos as `done`.

> Run this script once against production after deploying. It is safe to re-run — `ON CONFLICT DO NOTHING` means existing rows are untouched.

- [ ] **Step 1: Create seed script**

Create `scripts/pipelines/seed_lockedon_queue.py`:

```python
"""
One-time script to seed the lockedon_video_queue table.

Fetches all LockedOn Fantasy Basketball channel videos for the 2025-26 season,
inserts them into the queue, and marks already-ingested ones as 'done'.

Safe to re-run: ON CONFLICT DO NOTHING preserves existing rows.

Usage:
    python scripts/pipelines/seed_lockedon_queue.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from data.postgres.connection import PostgresConnection
from data.postgres.video_queue import VideoQueueManager
from data.postgres.vector_managers import YoutubeChannelVectorManager
from langchain_openai import OpenAIEmbeddings
from logger import get_logger, setup_logging

setup_logging(debug=False)
logger = get_logger(__name__)

CHANNEL = "@LockedOnFantasyBasketball"
SEASON_START = "2025-07-01"
SEASON_END = "2026-06-30"


def main():
    connection = PostgresConnection()
    embeddings = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))

    queue = VideoQueueManager(connection)
    queue.create_table()

    youtube_manager = YoutubeChannelVectorManager(
        connection=connection,
        embeddings=embeddings,
        chunk_size_seconds=30,
    )

    logger.info("Fetching existing ingested video IDs from vectorstore...")
    existing_ids = youtube_manager.get_existing_resource_ids()
    logger.info(f"  {len(existing_ids)} already ingested")

    logger.info(f"Fetching all videos from {CHANNEL} ({SEASON_START} → {SEASON_END})...")
    videos = youtube_manager.fetch_channel_videos(
        channel_handle=CHANNEL,
        published_after=SEASON_START,
        published_before=SEASON_END,
    )
    logger.info(f"  {len(videos)} videos found")

    inserted_pending = 0
    inserted_done = 0
    skipped = 0

    for video in videos:
        video_id = video["id"]
        publish_date = datetime.fromisoformat(
            video["publishedAt"].replace("Z", "+00:00")
        ).strftime("%Y-%m-%d")
        status = "done" if video_id in existing_ids else "pending"

        was_inserted = queue.upsert_video(
            video_id=video_id,
            title=video["title"],
            url=video["url"],
            publish_date=publish_date,
            status=status,
        )
        if was_inserted:
            if status == "done":
                inserted_done += 1
            else:
                inserted_pending += 1
        else:
            skipped += 1

    stats = queue.get_stats()
    logger.info("=" * 50)
    logger.info("Seed complete")
    logger.info(f"  Inserted pending : {inserted_pending}")
    logger.info(f"  Inserted done    : {inserted_done}")
    logger.info(f"  Skipped (exists) : {skipped}")
    logger.info(f"  Queue stats      : {stats}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run seed script locally against production DB**

Ensure `.env` has `DATABASE_URL` pointing to Railway Postgres (or use Railway's `railway run`):

```bash
env -u RAILWAY_TOKEN railway run python scripts/pipelines/seed_lockedon_queue.py
```

Expected output:
```
655 videos found
Seed complete
  Inserted pending : ~541
  Inserted done    : ~114
  Skipped (exists) : 0
  Queue stats      : {'pending': 541, 'done': 114}
```

- [ ] **Step 3: Verify queue in DB**

Run this SQL against Railway Postgres:

```sql
SELECT status, COUNT(*) FROM lockedon_video_queue GROUP BY status;
```

Expected: `pending` ~541, `done` ~114

- [ ] **Step 4: Commit**

```bash
git add scripts/pipelines/seed_lockedon_queue.py
git commit -m "feat: add seed script to populate lockedon_video_queue"
```

---

## Task 4: Rewrite update_lockedon_knowledge.py to Use Queue

**Files:**
- Modify: `scripts/pipelines/update_lockedon_knowledge.py`

Daily cron flow:
1. Check YouTube API for videos published yesterday → upsert as `pending` (new episodes auto-enqueued)
2. Pull up to `max_videos_added` pending rows from queue
3. For each: attempt transcript → on `AllTranscriptSourcesExhausted` mark failed + stop run; on success mark done; on other error mark failed + continue

- [ ] **Step 1: Rewrite the script**

Replace the full contents of `scripts/pipelines/update_lockedon_knowledge.py` with:

```python
"""
Daily cron: check for new LockedOn videos + process pending queue entries.

Run:
    python scripts/pipelines/update_lockedon_knowledge.py
"""

import os
import sys
import yaml
from datetime import datetime, timedelta
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from data.postgres.connection import PostgresConnection
from data.postgres.video_queue import AllTranscriptSourcesExhausted, VideoQueueManager
from data.postgres.vector_managers import YoutubeChannelVectorManager
from langchain_openai import OpenAIEmbeddings
from logger import get_logger, setup_logging

setup_logging(debug=False)
logger = get_logger(__name__)

vector_config_path = project_root / "config" / "vector_config.yaml"
vector_config = {}
if vector_config_path.exists():
    with open(vector_config_path) as f:
        vector_config = yaml.safe_load(f) or {}

rag_config_path = project_root / "config" / "rag_config.yaml"
rag_config = {}
if rag_config_path.exists():
    with open(rag_config_path) as f:
        rag_config = yaml.safe_load(f) or {}

CHANNEL = "@LockedOnFantasyBasketball"
DEV_MODE = os.environ.get("DEV_MODE", "false").lower() == "true"


def main():
    connection = PostgresConnection()
    embeddings = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))

    queue = VideoQueueManager(connection)
    queue.create_table()  # no-op if already exists

    youtube_manager = YoutubeChannelVectorManager(
        connection=connection,
        embeddings=embeddings,
        chunk_size_seconds=rag_config.get("youtube", {}).get("chunk_size_seconds", 30),
    )

    youtube_config = vector_config.get("youtube_channel", {})
    max_videos_added = 3 if DEV_MODE else youtube_config.get("max_videos_added", 15)
    delay_between_videos = youtube_config.get("delay_between_videos", 15.0)

    # Step 1: Enqueue any new videos published yesterday
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    logger.info(f"Checking for new videos published on {yesterday}...")
    try:
        new_videos = youtube_manager.fetch_channel_videos(
            channel_handle=CHANNEL,
            published_after=yesterday,
            published_before=yesterday,
        )
        for video in new_videos:
            publish_date = datetime.fromisoformat(
                video["publishedAt"].replace("Z", "+00:00")
            ).strftime("%Y-%m-%d")
            added = queue.upsert_video(
                video_id=video["id"],
                title=video["title"],
                url=video["url"],
                publish_date=publish_date,
            )
            if added:
                logger.info(f"  + Enqueued: {video['title']}")
    except Exception as e:
        logger.warning(f"  Could not check for new videos: {e}")

    # Step 2: Process pending videos
    pending = queue.get_pending(limit=max_videos_added)
    logger.info(f"\nProcessing {len(pending)} pending videos (max={max_videos_added})...")

    if not pending:
        logger.info("Queue is empty — nothing to do.")
        logger.info(f"Queue stats: {queue.get_stats()}")
        return

    videos_added = 0
    for i, video in enumerate(pending, 1):
        video_id = str(video["video_id"])
        title = video["title"] or video_id
        url = str(video["url"])
        publish_date = str(video["publish_date"]) if video["publish_date"] else None

        logger.info(f"\n[{i}/{len(pending)}] {title}")

        try:
            chunks_added = youtube_manager.add_resource_to_vectorstore(
                url=url,
                source_type="youtube",
                title=title,
                publish_date=publish_date,
                resource_id=video_id,
            )
            if chunks_added > 0:
                queue.mark_done(video_id)
                videos_added += 1
                logger.info(f"  ✓ Added {chunks_added} chunks")
            else:
                queue.mark_failed(video_id)
                logger.warning("  ✗ No transcript found")
        except AllTranscriptSourcesExhausted as e:
            queue.mark_failed(video_id)
            logger.error(f"  ✗ All sources exhausted: {e}")
            logger.error("  Stopping run — will retry tomorrow.")
            break
        except Exception as e:
            queue.mark_failed(video_id)
            logger.error(f"  ✗ Failed: {e}")

        if i < len(pending) and delay_between_videos > 0:
            import time
            time.sleep(delay_between_videos)

    logger.info("\n" + "=" * 50)
    logger.info(f"Videos added this run : {videos_added}")
    logger.info(f"Queue stats           : {queue.get_stats()}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run existing tests to confirm nothing broken**

```bash
pytest tests/ -v
```

Expected: all previously passing tests still PASS

- [ ] **Step 3: Verify script runs locally (dry-run check)**

With `DEV_MODE=true` and a local `.env`, run:

```bash
DEV_MODE=true python scripts/pipelines/update_lockedon_knowledge.py
```

Expected log output:
```
Checking for new videos published on <yesterday>...
Processing N pending videos (max=3)...
[1/3] <title>
  ⚡ Circuit breaker open — skipping YouTube, going straight to Supadata
  ✗ Supadata fallback failed: ...
  ✗ All sources exhausted: ...
  Stopping run — will retry tomorrow.
==============================
Videos added this run : 0
Queue stats           : {'pending': 541, 'done': 114, 'failed': 1}
```

- [ ] **Step 4: Commit**

```bash
git add scripts/pipelines/update_lockedon_knowledge.py
git commit -m "feat: rewrite lockedon pipeline to use VideoQueueManager"
```

---

## Task 5: Push and Verify in Production

- [ ] **Step 1: Push to origin**

```bash
git push origin main
```

Railway auto-deploys on push to `main`.

- [ ] **Step 2: Run seed script against production**

```bash
env -u RAILWAY_TOKEN railway run python scripts/pipelines/seed_lockedon_queue.py
```

Confirm queue stats show ~541 pending, ~114 done.

- [ ] **Step 3: Manually trigger a cron run to verify**

```bash
env -u RAILWAY_TOKEN railway run python scripts/pipelines/update_lockedon_knowledge.py
```

Confirm logs show:
- New video check runs
- Pending videos attempted
- Run stops cleanly on `AllTranscriptSourcesExhausted` (until YouTube unblocks)
- Queue stats update correctly

- [ ] **Step 4: Verify queue state in DB**

```sql
SELECT status, COUNT(*) FROM lockedon_video_queue GROUP BY status;
SELECT video_id, title, status, attempts, last_attempted_at
FROM lockedon_video_queue
WHERE status = 'failed'
ORDER BY last_attempted_at DESC
LIMIT 5;
```

Expected: failed videos have `attempts = 1`, `last_attempted_at` set to now.
