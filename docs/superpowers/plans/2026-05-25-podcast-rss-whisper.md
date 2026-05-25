# Podcast RSS + Whisper Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ingest fantasy basketball podcast episodes by fetching audio via RSS feed, transcribing with OpenAI Whisper, removing non-content segments (ads, intros, outros) via `gpt-4o-mini`, then chunking and embedding clean transcripts into pgvector.

**Architecture:** New `PodcastVectorManager` extends `BaseVectorManager`. Pipeline: RSS fetch (feedparser) → download audio (streaming to temp file) → Whisper transcription (optimized with domain prompt + `language="en"`) → LLM segment cleaning (`gpt-4o-mini` identifies and drops ad/intro/outro segments) → time-window chunking at 120s on clean segments → pgvector embed. New `update_podcast_knowledge.py` pipeline registered in the master `update_vector_database.py` cron.

**Tech Stack:** `feedparser` (already in `requirements.txt`), `openai` Whisper API + `gpt-4o-mini` (same `OPENAI_API_KEY`), `requests` (streaming download), `tempfile`, LangChain `Document`, PostgreSQL pgvector.

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `data/postgres/vector_managers/podcast.py` | `PodcastVectorManager` — RSS fetch, download, Whisper, LLM clean, chunk |
| Modify | `data/postgres/vector_managers/__init__.py` | Export `PodcastVectorManager` |
| Modify | `config/vector_config.yaml` | Add `podcast` section |
| Create | `scripts/pipelines/update_podcast_knowledge.py` | Pipeline script |
| Modify | `scripts/pipelines/update_vector_database.py` | Add podcast to master cron |
| Create | `tests/test_podcast_vector_manager.py` | Unit tests — all external calls mocked |

---

## Task 1: Add podcast config to `vector_config.yaml`

**Files:**
- Modify: `config/vector_config.yaml`

- [ ] **Step 1: Add podcast section**

Append to `config/vector_config.yaml`:

```yaml
podcast:
  feeds:
    - name: "LockedOn Fantasy Basketball"
      rss_url: "https://feeds.megaphone.fm/LKO7065672669"  # verify from your podcast app or rss.com
  max_episodes_added: 5      # max new episodes to add per cron run
  chunk_size_seconds: 120    # 2-minute windows; podcast topics span 2-3 min
  delay_between_episodes: 2.0
```

> **Note:** Verify the RSS URL by opening LockedOn Fantasy Basketball in any podcast app → share → copy RSS link. Replace `rss_url` before running.

- [ ] **Step 2: Commit**

```bash
git add config/vector_config.yaml
git commit -m "config: add podcast RSS ingestion settings"
```

---

## Task 2: Write failing tests for `_chunk_segments`

**Files:**
- Create: `data/postgres/vector_managers/podcast.py` (skeleton)
- Create: `tests/test_podcast_vector_manager.py`

- [ ] **Step 1: Create skeleton `podcast.py`**

Create `data/postgres/vector_managers/podcast.py`:

```python
from __future__ import annotations

import os
import time
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from data.postgres.connection import PostgresConnection
from data.postgres.vector_managers.base import BaseVectorManager
from data.data_class import YouTubeVideo
from logger import get_logger

logger = get_logger(__name__)

WHISPER_PROMPT = (
    "Fantasy basketball podcast. NBA players: LeBron James, Nikola Jokić, "
    "Shai Gilgeous-Alexander, Anthony Edwards, Victor Wembanyama, Ja Morant, "
    "Jayson Tatum, Luka Dončić, Stephen Curry, Giannis Antetokounmpo. "
    "Terms: waiver wire, trade deadline, injury report, day-to-day, BPM, VORP, "
    "True Shooting, PER, Points Per Game, rebounds, assists, blocks, steals."
)


class PodcastVectorManager(BaseVectorManager):

    def __init__(
        self,
        connection: PostgresConnection,
        embeddings: OpenAIEmbeddings,
        table_name: Optional[str] = None,
        chunk_size_seconds: int = 120,
    ):
        super().__init__(connection, embeddings, table_name)
        self.chunk_size_seconds = chunk_size_seconds

    def load_resources_from_yaml(self, vectordata_yaml=None) -> List[Tuple[str, YouTubeVideo]]:
        return []

    def load_resource_content(self, url, source_type='podcast', title=None,
                              upload_date=None, publish_date=None,
                              resource_id=None, chunk_size_seconds=None, **kwargs) -> List[Document]:
        raise NotImplementedError

    def add_resource_to_vectorstore(self, url, source_type='podcast', title=None,
                                    upload_date=None, publish_date=None,
                                    resource_id=None, chunk_size_seconds=None, **kwargs) -> int:
        raise NotImplementedError

    def _chunk_segments(
        self,
        segments: List[Dict],
        chunk_size_seconds: int,
        audio_url: str,
        title: Optional[str],
        publish_date: Optional[str],
        resource_id: Optional[str],
        source_type: str,
    ) -> List[Document]:
        raise NotImplementedError

    def _make_document(
        self,
        chunk_text: List[str],
        start_seconds: float,
        audio_url: str,
        title: Optional[str],
        publish_date: Optional[str],
        resource_id: Optional[str],
        source_type: str,
    ) -> Document:
        raise NotImplementedError

    def fetch_feed_episodes(self, rss_url: str, max_episodes: Optional[int] = None,
                            season_start: Optional[str] = None,
                            season_end: Optional[str] = None) -> List[Dict]:
        raise NotImplementedError

    def _download_audio(self, audio_url: str) -> Path:
        raise NotImplementedError

    def _transcribe_audio(self, audio_path: Path) -> List[Dict]:
        raise NotImplementedError

    def _clean_segments(self, segments: List[Dict]) -> List[Dict]:
        raise NotImplementedError

    def add_feed_to_vectorstore(self, rss_url: str, feed_name: str = 'podcast',
                                max_episodes_added: Optional[int] = None,
                                season_start: Optional[str] = None,
                                season_end: Optional[str] = None,
                                chunk_size_seconds: Optional[int] = None,
                                skip_existing: bool = True,
                                delay_between_episodes: float = 2.0) -> Dict:
        raise NotImplementedError
```

- [ ] **Step 2: Write tests for `_chunk_segments`**

Create `tests/test_podcast_vector_manager.py`:

```python
"""Unit tests for PodcastVectorManager."""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from data.postgres.vector_managers.podcast import PodcastVectorManager


@pytest.fixture
def manager():
    mock_conn = MagicMock()
    mock_conn.get_engine.return_value = MagicMock()
    with patch('data.postgres.vector_managers.base.setup_pgvector_extension'), \
         patch('data.postgres.vector_managers.base.PGVector'):
        return PodcastVectorManager(
            connection=mock_conn,
            embeddings=MagicMock(),
            table_name='test_table',
            chunk_size_seconds=30,
        )


class TestChunkSegments:

    def test_single_chunk_when_all_segments_fit(self, manager):
        segments = [
            {"start": 0.0, "end": 10.0, "text": "Hello"},
            {"start": 10.0, "end": 20.0, "text": "world"},
        ]
        docs = manager._chunk_segments(
            segments, chunk_size_seconds=30,
            audio_url="http://example.com/ep.mp3",
            title="Episode 1", publish_date="2026-01-01",
            resource_id="ep-001", source_type="podcast",
        )
        assert len(docs) == 1
        assert "Hello" in docs[0].page_content
        assert "world" in docs[0].page_content

    def test_splits_into_multiple_chunks(self, manager):
        segments = [
            {"start": 0.0,  "end": 15.0, "text": "A"},
            {"start": 15.0, "end": 30.0, "text": "B"},
            {"start": 30.0, "end": 45.0, "text": "C"},
            {"start": 45.0, "end": 60.0, "text": "D"},
        ]
        docs = manager._chunk_segments(
            segments, chunk_size_seconds=30,
            audio_url="http://example.com/ep.mp3",
            title="Episode 1", publish_date="2026-01-01",
            resource_id="ep-001", source_type="podcast",
        )
        assert len(docs) == 2
        assert "A" in docs[0].page_content and "B" in docs[0].page_content
        assert "C" in docs[1].page_content and "D" in docs[1].page_content

    def test_chunk_metadata_fields(self, manager):
        segments = [{"start": 0.0, "end": 10.0, "text": "Hello"}]
        docs = manager._chunk_segments(
            segments, chunk_size_seconds=30,
            audio_url="http://example.com/ep.mp3",
            title="My Episode", publish_date="2026-01-15",
            resource_id="ep-42", source_type="podcast",
        )
        meta = docs[0].metadata
        assert meta["source_type"] == "podcast"
        assert meta["resource_id"] == "ep-42"
        assert meta["title"] == "My Episode"
        assert meta["publish_date"] == "2026-01-15"
        assert "source" in meta
        assert "start_seconds" in meta

    def test_empty_segments_returns_empty_list(self, manager):
        docs = manager._chunk_segments(
            [], chunk_size_seconds=30,
            audio_url="http://example.com/ep.mp3",
            title=None, publish_date=None,
            resource_id=None, source_type="podcast",
        )
        assert docs == []

    def test_skips_empty_text_segments(self, manager):
        segments = [
            {"start": 0.0,  "end": 5.0,  "text": ""},
            {"start": 5.0,  "end": 10.0, "text": "  "},
            {"start": 10.0, "end": 20.0, "text": "Real content"},
        ]
        docs = manager._chunk_segments(
            segments, chunk_size_seconds=30,
            audio_url="http://example.com/ep.mp3",
            title=None, publish_date=None,
            resource_id=None, source_type="podcast",
        )
        assert len(docs) == 1
        assert "Real content" in docs[0].page_content
```

- [ ] **Step 3: Run to confirm tests fail**

```bash
cd /Users/matt/Code/ClutchAI && python -m pytest tests/test_podcast_vector_manager.py::TestChunkSegments -v 2>&1 | tail -20
```

Expected: FAIL with `NotImplementedError`

---

## Task 3: Implement `_chunk_segments` and `_make_document`

**Files:**
- Modify: `data/postgres/vector_managers/podcast.py`

- [ ] **Step 1: Implement both methods**

Replace the two `raise NotImplementedError` stubs with:

```python
def _chunk_segments(
    self,
    segments: List[Dict],
    chunk_size_seconds: int,
    audio_url: str,
    title: Optional[str],
    publish_date: Optional[str],
    resource_id: Optional[str],
    source_type: str,
) -> List[Document]:
    docs = []
    chunk_text: List[str] = []
    chunk_start: Optional[float] = None

    for segment in segments:
        text = segment.get("text", "").strip()
        if not text:
            continue
        start = segment.get("start", 0.0)
        if chunk_start is None:
            chunk_start = start

        if (start - chunk_start) >= chunk_size_seconds and chunk_text:
            docs.append(self._make_document(
                chunk_text, chunk_start, audio_url,
                title, publish_date, resource_id, source_type,
            ))
            chunk_text = [text]
            chunk_start = start
        else:
            chunk_text.append(text)

    if chunk_text and chunk_start is not None:
        docs.append(self._make_document(
            chunk_text, chunk_start, audio_url,
            title, publish_date, resource_id, source_type,
        ))

    return docs

def _make_document(
    self,
    chunk_text: List[str],
    start_seconds: float,
    audio_url: str,
    title: Optional[str],
    publish_date: Optional[str],
    resource_id: Optional[str],
    source_type: str,
) -> Document:
    meta = {
        "source": f"{audio_url}#t={int(start_seconds)}s",
        "source_type": source_type,
        "start_seconds": start_seconds,
    }
    if resource_id:
        meta["resource_id"] = resource_id
    if title:
        meta["title"] = title
    if publish_date:
        meta["publish_date"] = publish_date
    return Document(page_content=" ".join(chunk_text), metadata=meta)
```

- [ ] **Step 2: Run tests to confirm they pass**

```bash
cd /Users/matt/Code/ClutchAI && python -m pytest tests/test_podcast_vector_manager.py::TestChunkSegments -v 2>&1 | tail -20
```

Expected: All 5 PASS

- [ ] **Step 3: Commit**

```bash
git add data/postgres/vector_managers/podcast.py tests/test_podcast_vector_manager.py
git commit -m "feat: add PodcastVectorManager skeleton and _chunk_segments with tests"
```

---

## Task 4: Write failing tests for `_clean_segments` → implement → pass

`_clean_segments` calls `gpt-4o-mini` with the full segment list and asks it to return indices of non-content segments (ads, sponsor reads, intros, outros, social plugs) to remove.

**Files:**
- Modify: `tests/test_podcast_vector_manager.py`
- Modify: `data/postgres/vector_managers/podcast.py`

- [ ] **Step 1: Add tests for `_clean_segments`**

Append to `tests/test_podcast_vector_manager.py`:

```python
class TestCleanSegments:

    def _make_segments(self):
        return [
            {"start": 0.0,  "end": 10.0, "text": "Welcome to LockedOn Fantasy Basketball!"},
            {"start": 10.0, "end": 20.0, "text": "Today's show is brought to you by DraftKings."},
            {"start": 20.0, "end": 30.0, "text": "Use promo code LOCKED for 20% off."},
            {"start": 30.0, "end": 40.0, "text": "Alright, let's talk about Nikola Jokic."},
            {"start": 40.0, "end": 50.0, "text": "His numbers this week have been elite."},
            {"start": 50.0, "end": 60.0, "text": "Follow us on Twitter @LockedOnFantasy."},
        ]

    def _mock_openai_response(self, indices_to_remove):
        """Build a mock OpenAI chat completion that returns a JSON remove list."""
        import json
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({"remove": indices_to_remove})
        return mock_response

    def test_removes_ad_and_intro_segments(self, manager):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_openai_response([0, 1, 2, 5])
        with patch('data.postgres.vector_managers.podcast.OpenAI', return_value=mock_client):
            clean = manager._clean_segments(self._make_segments())
        texts = [s["text"] for s in clean]
        assert "Alright, let's talk about Nikola Jokic." in texts
        assert "His numbers this week have been elite." in texts
        assert "Today's show is brought to you by DraftKings." not in texts
        assert "Use promo code LOCKED for 20% off." not in texts

    def test_returns_all_segments_when_nothing_to_remove(self, manager):
        segments = [
            {"start": 0.0, "end": 10.0, "text": "Great analysis today."},
            {"start": 10.0, "end": 20.0, "text": "Jokic drops 40 points."},
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_openai_response([])
        with patch('data.postgres.vector_managers.podcast.OpenAI', return_value=mock_client):
            clean = manager._clean_segments(segments)
        assert len(clean) == 2

    def test_returns_original_segments_on_llm_error(self, manager):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API error")
        with patch('data.postgres.vector_managers.podcast.OpenAI', return_value=mock_client):
            clean = manager._clean_segments(self._make_segments())
        assert len(clean) == len(self._make_segments())

    def test_returns_original_on_invalid_json(self, manager):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "not valid json"
        mock_client.chat.completions.create.return_value = mock_response
        with patch('data.postgres.vector_managers.podcast.OpenAI', return_value=mock_client):
            clean = manager._clean_segments(self._make_segments())
        assert len(clean) == len(self._make_segments())
```

- [ ] **Step 2: Run to confirm they fail**

```bash
cd /Users/matt/Code/ClutchAI && python -m pytest tests/test_podcast_vector_manager.py::TestCleanSegments -v 2>&1 | tail -20
```

Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement `_clean_segments`**

Add `import json` and `from openai import OpenAI` to the top-level imports in `podcast.py`. Then replace the stub:

```python
def _clean_segments(self, segments: List[Dict]) -> List[Dict]:
    if not segments:
        return segments

    numbered = "\n".join(
        f"[{i}] {seg.get('text', '').strip()}"
        for i, seg in enumerate(segments)
    )
    prompt = (
        "You are filtering a fantasy basketball podcast transcript. "
        "Identify segment indices that are: advertisements, sponsor reads, "
        "promo codes, show introductions, outros, social media promotions, "
        "or any filler unrelated to basketball analysis. "
        "Return ONLY a JSON object with key 'remove' containing a list of integer indices. "
        "Do NOT remove any basketball analysis, player discussion, trade talk, or injury news. "
        "Example: {\"remove\": [0, 1, 5]}\n\n"
        f"Segments:\n{numbered}"
    )

    try:
        client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        indices_to_remove = set(result.get("remove", []))
        kept = [seg for i, seg in enumerate(segments) if i not in indices_to_remove]
        removed_count = len(segments) - len(kept)
        logger.info(f"  Cleaned transcript: removed {removed_count} non-content segments "
                    f"({len(kept)}/{len(segments)} kept)")
        return kept
    except Exception as e:
        logger.warning(f"  Transcript cleaning failed ({e}), using raw segments")
        return segments
```

- [ ] **Step 4: Run to confirm they pass**

```bash
cd /Users/matt/Code/ClutchAI && python -m pytest tests/test_podcast_vector_manager.py::TestCleanSegments -v 2>&1 | tail -20
```

Expected: All 4 PASS

- [ ] **Step 5: Commit**

```bash
git add data/postgres/vector_managers/podcast.py tests/test_podcast_vector_manager.py
git commit -m "feat: implement _clean_segments with gpt-4o-mini ad/intro removal and tests"
```

---

## Task 5: Write failing tests for `fetch_feed_episodes` → implement → pass

**Files:**
- Modify: `tests/test_podcast_vector_manager.py`
- Modify: `data/postgres/vector_managers/podcast.py`

- [ ] **Step 1: Add tests**

Append to `tests/test_podcast_vector_manager.py`:

```python
import time as time_mod


def _make_entry(guid, title, audio_url, episode_url, year=2026, month=1, day=15):
    entry = MagicMock()
    entry.id = guid
    entry.title = title
    entry.enclosures = [MagicMock(href=audio_url)]
    entry.link = episode_url
    entry.published_parsed = time_mod.struct_time(
        (year, month, day, 12, 0, 0, 0, 15, 0)
    )
    return entry


class TestFetchFeedEpisodes:

    def test_returns_episode_list(self, manager):
        mock_feed = MagicMock()
        mock_feed.entries = [
            _make_entry("guid-1", "Episode 1", "http://cdn.example.com/ep1.mp3",
                        "http://example.com/ep1"),
        ]
        with patch('feedparser.parse', return_value=mock_feed):
            episodes = manager.fetch_feed_episodes("http://feeds.example.com/rss")
        assert len(episodes) == 1
        ep = episodes[0]
        assert ep["id"] == "guid-1"
        assert ep["title"] == "Episode 1"
        assert ep["audio_url"] == "http://cdn.example.com/ep1.mp3"
        assert ep["episode_url"] == "http://example.com/ep1"
        assert ep["publish_date"] == "2026-01-15"

    def test_filters_by_season_start(self, manager):
        mock_feed = MagicMock()
        mock_feed.entries = [
            _make_entry("guid-old", "Old", "http://cdn.example.com/old.mp3",
                        "http://example.com/old", year=2024, month=6, day=1),
            _make_entry("guid-new", "New", "http://cdn.example.com/new.mp3",
                        "http://example.com/new", year=2026, month=1, day=15),
        ]
        with patch('feedparser.parse', return_value=mock_feed):
            episodes = manager.fetch_feed_episodes(
                "http://feeds.example.com/rss", season_start="2025-07-01"
            )
        assert len(episodes) == 1
        assert episodes[0]["id"] == "guid-new"

    def test_respects_max_episodes(self, manager):
        mock_feed = MagicMock()
        mock_feed.entries = [
            _make_entry(f"guid-{i}", f"Episode {i}", f"http://cdn.example.com/ep{i}.mp3",
                        f"http://example.com/ep{i}", day=i)
            for i in range(1, 11)
        ]
        with patch('feedparser.parse', return_value=mock_feed):
            episodes = manager.fetch_feed_episodes(
                "http://feeds.example.com/rss", max_episodes=3
            )
        assert len(episodes) == 3

    def test_skips_entries_without_enclosures(self, manager):
        entry = MagicMock()
        entry.id = "guid-no-audio"
        entry.title = "No Audio"
        entry.enclosures = []
        entry.link = "http://example.com/ep"
        entry.published_parsed = time_mod.struct_time((2026, 1, 15, 12, 0, 0, 0, 15, 0))
        mock_feed = MagicMock()
        mock_feed.entries = [entry]
        with patch('feedparser.parse', return_value=mock_feed):
            episodes = manager.fetch_feed_episodes("http://feeds.example.com/rss")
        assert episodes == []
```

- [ ] **Step 2: Run to confirm they fail**

```bash
cd /Users/matt/Code/ClutchAI && python -m pytest tests/test_podcast_vector_manager.py::TestFetchFeedEpisodes -v 2>&1 | tail -20
```

Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement `fetch_feed_episodes`**

Add imports at top of `podcast.py`:
```python
import feedparser
from datetime import datetime
```

Replace the stub:

```python
def fetch_feed_episodes(
    self,
    rss_url: str,
    max_episodes: Optional[int] = None,
    season_start: Optional[str] = None,
    season_end: Optional[str] = None,
) -> List[Dict]:
    feed = feedparser.parse(rss_url)
    season_start_dt = datetime.fromisoformat(season_start) if season_start else None
    season_end_dt = datetime.fromisoformat(season_end) if season_end else None

    episodes = []
    for entry in feed.entries:
        if not getattr(entry, 'enclosures', None):
            continue
        audio_url = entry.enclosures[0].href if entry.enclosures else None
        if not audio_url:
            continue

        publish_dt = (
            datetime.fromtimestamp(time.mktime(entry.published_parsed))
            if getattr(entry, 'published_parsed', None) else None
        )
        if season_start_dt and publish_dt and publish_dt < season_start_dt:
            continue
        if season_end_dt and publish_dt and publish_dt > season_end_dt:
            continue

        episodes.append({
            "id": getattr(entry, 'id', None) or entry.link,
            "title": entry.title,
            "audio_url": audio_url,
            "episode_url": entry.link,
            "publish_date": publish_dt.strftime('%Y-%m-%d') if publish_dt else None,
        })

        if max_episodes and len(episodes) >= max_episodes:
            break

    return episodes
```

- [ ] **Step 4: Run to confirm they pass**

```bash
cd /Users/matt/Code/ClutchAI && python -m pytest tests/test_podcast_vector_manager.py::TestFetchFeedEpisodes -v 2>&1 | tail -20
```

Expected: All 4 PASS

- [ ] **Step 5: Commit**

```bash
git add data/postgres/vector_managers/podcast.py tests/test_podcast_vector_manager.py
git commit -m "feat: implement fetch_feed_episodes with season filtering and tests"
```

---

## Task 6: Implement `load_resource_content` (full pipeline: download → transcribe → clean → chunk)

**Files:**
- Modify: `tests/test_podcast_vector_manager.py`
- Modify: `data/postgres/vector_managers/podcast.py`

- [ ] **Step 1: Add tests**

Append to `tests/test_podcast_vector_manager.py`:

```python
class TestLoadResourceContent:

    def test_returns_documents_with_correct_metadata(self, manager):
        mock_segments = [
            {"start": 0.0,  "end": 20.0, "text": "Welcome to the show"},
            {"start": 20.0, "end": 40.0, "text": "Today we talk basketball"},
        ]
        with patch.object(manager, '_download_audio', return_value=Path('/tmp/ep.mp3')), \
             patch.object(manager, '_transcribe_audio', return_value=mock_segments), \
             patch.object(manager, '_clean_segments', return_value=mock_segments), \
             patch('pathlib.Path.unlink'):
            docs = manager.load_resource_content(
                url="http://cdn.example.com/ep.mp3",
                source_type="podcast",
                title="Episode 42",
                publish_date="2026-01-15",
                resource_id="ep-42",
                chunk_size_seconds=30,
            )
        assert len(docs) == 1
        assert docs[0].metadata["title"] == "Episode 42"
        assert docs[0].metadata["resource_id"] == "ep-42"
        assert docs[0].metadata["source_type"] == "podcast"

    def test_cleans_up_temp_file_on_success(self, manager):
        mock_path = MagicMock(spec=Path)
        segments = [{"start": 0.0, "end": 10.0, "text": "hello"}]
        with patch.object(manager, '_download_audio', return_value=mock_path), \
             patch.object(manager, '_transcribe_audio', return_value=segments), \
             patch.object(manager, '_clean_segments', return_value=segments):
            manager.load_resource_content(url="http://cdn.example.com/ep.mp3",
                                          source_type="podcast", resource_id="ep-1")
        mock_path.unlink.assert_called_once_with(missing_ok=True)

    def test_cleans_up_temp_file_on_error(self, manager):
        mock_path = MagicMock(spec=Path)
        with patch.object(manager, '_download_audio', return_value=mock_path), \
             patch.object(manager, '_transcribe_audio', side_effect=RuntimeError("API error")):
            with pytest.raises(RuntimeError):
                manager.load_resource_content(url="http://cdn.example.com/ep.mp3",
                                              source_type="podcast", resource_id="ep-1")
        mock_path.unlink.assert_called_once_with(missing_ok=True)

    def test_calls_clean_segments_between_transcribe_and_chunk(self, manager):
        raw_segments = [{"start": 0.0, "end": 10.0, "text": "ad content"}]
        clean_segments = [{"start": 10.0, "end": 20.0, "text": "real content"}]
        with patch.object(manager, '_download_audio', return_value=Path('/tmp/ep.mp3')), \
             patch.object(manager, '_transcribe_audio', return_value=raw_segments), \
             patch.object(manager, '_clean_segments', return_value=clean_segments) as mock_clean, \
             patch('pathlib.Path.unlink'):
            docs = manager.load_resource_content(url="http://cdn.example.com/ep.mp3",
                                                  source_type="podcast")
        mock_clean.assert_called_once_with(raw_segments)
        assert "real content" in docs[0].page_content
```

- [ ] **Step 2: Run to confirm they fail**

```bash
cd /Users/matt/Code/ClutchAI && python -m pytest tests/test_podcast_vector_manager.py::TestLoadResourceContent -v 2>&1 | tail -20
```

Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement `_download_audio`, `_transcribe_audio`, `load_resource_content`, `add_resource_to_vectorstore`**

Add `import requests` to top-level imports in `podcast.py`. Replace all four stubs:

```python
def _download_audio(self, audio_url: str) -> Path:
    response = requests.get(audio_url, stream=True, timeout=60)
    response.raise_for_status()
    suffix = Path(audio_url.split('?')[0]).suffix or '.mp3'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        for chunk in response.iter_content(chunk_size=8192):
            tmp.write(chunk)
    finally:
        tmp.close()
    return Path(tmp.name)

def _transcribe_audio(self, audio_path: Path) -> List[Dict]:
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    if file_size_mb > 24:
        raise ValueError(
            f"Audio file is {file_size_mb:.1f} MB — exceeds Whisper's 25 MB limit. "
            "Episode skipped."
        )
    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
    with open(audio_path, 'rb') as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="en",
            prompt=WHISPER_PROMPT,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
            temperature=0,
        )
    return [
        {"start": seg.start, "end": seg.end, "text": seg.text}
        for seg in response.segments
    ]

def load_resource_content(
    self,
    url: str,
    source_type: str = 'podcast',
    title: Optional[str] = None,
    upload_date: Optional[str] = None,
    publish_date: Optional[str] = None,
    resource_id: Optional[str] = None,
    chunk_size_seconds: Optional[int] = None,
    **kwargs,
) -> List[Document]:
    chunk_size = chunk_size_seconds if chunk_size_seconds is not None else self.chunk_size_seconds
    audio_path = self._download_audio(url)
    try:
        raw_segments = self._transcribe_audio(audio_path)
    finally:
        audio_path.unlink(missing_ok=True)

    clean_segs = self._clean_segments(raw_segments)
    return self._chunk_segments(
        segments=clean_segs,
        chunk_size_seconds=chunk_size,
        audio_url=url,
        title=title,
        publish_date=publish_date or upload_date,
        resource_id=resource_id,
        source_type=source_type,
    )

def add_resource_to_vectorstore(
    self,
    url: str,
    source_type: str = 'podcast',
    title: Optional[str] = None,
    upload_date: Optional[str] = None,
    publish_date: Optional[str] = None,
    resource_id: Optional[str] = None,
    chunk_size_seconds: Optional[int] = None,
    **kwargs,
) -> int:
    docs = self.load_resource_content(
        url=url, source_type=source_type, title=title,
        upload_date=upload_date, publish_date=publish_date,
        resource_id=resource_id, chunk_size_seconds=chunk_size_seconds,
        **kwargs,
    )
    if not docs:
        return 0
    self._add_documents_to_vectorstore(docs)
    return len(docs)
```

- [ ] **Step 4: Run to confirm they pass**

```bash
cd /Users/matt/Code/ClutchAI && python -m pytest tests/test_podcast_vector_manager.py::TestLoadResourceContent -v 2>&1 | tail -20
```

Expected: All 4 PASS

- [ ] **Step 5: Commit**

```bash
git add data/postgres/vector_managers/podcast.py tests/test_podcast_vector_manager.py
git commit -m "feat: implement full transcription pipeline (download→Whisper→clean→chunk) with tests"
```

---

## Task 7: Implement `add_feed_to_vectorstore` with tests

**Files:**
- Modify: `tests/test_podcast_vector_manager.py`
- Modify: `data/postgres/vector_managers/podcast.py`

- [ ] **Step 1: Add tests**

Append to `tests/test_podcast_vector_manager.py`:

```python
class TestAddFeedToVectorstore:

    def _sample_episodes(self):
        return [
            {"id": "ep-1", "title": "Ep 1", "audio_url": "http://cdn.example.com/ep1.mp3",
             "episode_url": "http://example.com/ep1", "publish_date": "2026-01-10"},
            {"id": "ep-2", "title": "Ep 2", "audio_url": "http://cdn.example.com/ep2.mp3",
             "episode_url": "http://example.com/ep2", "publish_date": "2026-01-17"},
        ]

    def test_adds_new_episodes(self, manager):
        with patch.object(manager, 'fetch_feed_episodes', return_value=self._sample_episodes()), \
             patch.object(manager, 'get_existing_resource_ids', return_value=set()), \
             patch.object(manager, 'add_resource_to_vectorstore', return_value=10):
            results = manager.add_feed_to_vectorstore(rss_url="http://feeds.example.com/rss",
                                                       feed_name="Test Podcast")
        assert results["episodes_added"] == 2
        assert results["chunks_added"] == 20

    def test_skips_existing_episodes(self, manager):
        with patch.object(manager, 'fetch_feed_episodes', return_value=self._sample_episodes()), \
             patch.object(manager, 'get_existing_resource_ids', return_value={"ep-1"}), \
             patch.object(manager, 'add_resource_to_vectorstore', return_value=10):
            results = manager.add_feed_to_vectorstore(rss_url="http://feeds.example.com/rss",
                                                       feed_name="Test Podcast",
                                                       skip_existing=True)
        assert results["episodes_added"] == 1
        assert results["episodes_skipped"] == 1

    def test_respects_max_episodes_added(self, manager):
        with patch.object(manager, 'fetch_feed_episodes', return_value=self._sample_episodes()), \
             patch.object(manager, 'get_existing_resource_ids', return_value=set()), \
             patch.object(manager, 'add_resource_to_vectorstore', return_value=10):
            results = manager.add_feed_to_vectorstore(rss_url="http://feeds.example.com/rss",
                                                       feed_name="Test Podcast",
                                                       max_episodes_added=1)
        assert results["episodes_added"] == 1

    def test_records_failed_episodes(self, manager):
        with patch.object(manager, 'fetch_feed_episodes', return_value=self._sample_episodes()), \
             patch.object(manager, 'get_existing_resource_ids', return_value=set()), \
             patch.object(manager, 'add_resource_to_vectorstore',
                          side_effect=RuntimeError("download failed")):
            results = manager.add_feed_to_vectorstore(rss_url="http://feeds.example.com/rss",
                                                       feed_name="Test Podcast")
        assert results["episodes_failed"] == 2
        assert results["episodes_added"] == 0
```

- [ ] **Step 2: Run to confirm they fail**

```bash
cd /Users/matt/Code/ClutchAI && python -m pytest tests/test_podcast_vector_manager.py::TestAddFeedToVectorstore -v 2>&1 | tail -20
```

Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement `add_feed_to_vectorstore`**

Replace the stub:

```python
def add_feed_to_vectorstore(
    self,
    rss_url: str,
    feed_name: str = 'podcast',
    max_episodes_added: Optional[int] = None,
    season_start: Optional[str] = None,
    season_end: Optional[str] = None,
    chunk_size_seconds: Optional[int] = None,
    skip_existing: bool = True,
    delay_between_episodes: float = 2.0,
) -> Dict:
    chunk_size = chunk_size_seconds if chunk_size_seconds is not None else self.chunk_size_seconds
    logger.info(f"Fetching episodes from: {feed_name} ({rss_url})")
    episodes = self.fetch_feed_episodes(rss_url, season_start=season_start, season_end=season_end)
    logger.info(f"Found {len(episodes)} episodes in date range")

    existing_ids = self.get_existing_resource_ids() if skip_existing else set()
    results = {
        "episodes_found": len(episodes),
        "episodes_added": 0,
        "episodes_skipped": 0,
        "episodes_failed": 0,
        "chunks_added": 0,
    }

    for i, episode in enumerate(episodes, 1):
        ep_id = episode["id"]
        if skip_existing and ep_id in existing_ids:
            logger.info(f"  [{i}/{len(episodes)}] ⏭ Skipping: {episode['title']}")
            results["episodes_skipped"] += 1
            continue

        logger.info(f"  [{i}/{len(episodes)}] Processing: {episode['title']}")
        try:
            chunks = self.add_resource_to_vectorstore(
                url=episode["audio_url"],
                source_type="podcast",
                title=episode["title"],
                publish_date=episode.get("publish_date"),
                resource_id=ep_id,
                chunk_size_seconds=chunk_size,
            )
            results["episodes_added"] += 1
            results["chunks_added"] += chunks
            existing_ids.add(ep_id)
            logger.info(f"    ✓ Added {chunks} chunks")

            if max_episodes_added and results["episodes_added"] >= max_episodes_added:
                logger.info(f"  Reached limit of {max_episodes_added} episodes. Stopping.")
                break
        except Exception as e:
            results["episodes_failed"] += 1
            logger.error(f"    ✗ Failed: {e}")

        if i < len(episodes) and delay_between_episodes > 0:
            if max_episodes_added is None or results["episodes_added"] < max_episodes_added:
                time.sleep(delay_between_episodes)

    logger.info(f"Complete — added: {results['episodes_added']}, "
                f"skipped: {results['episodes_skipped']}, "
                f"failed: {results['episodes_failed']}, "
                f"chunks: {results['chunks_added']}")
    return results
```

- [ ] **Step 4: Run all tests to confirm everything passes**

```bash
cd /Users/matt/Code/ClutchAI && python -m pytest tests/test_podcast_vector_manager.py -v 2>&1 | tail -30
```

Expected: All 17 tests PASS

- [ ] **Step 5: Commit**

```bash
git add data/postgres/vector_managers/podcast.py tests/test_podcast_vector_manager.py
git commit -m "feat: implement add_feed_to_vectorstore with skip/limit/error handling and tests"
```

---

## Task 8: Wire up exports, pipeline script, and master cron

**Files:**
- Modify: `data/postgres/vector_managers/__init__.py`
- Create: `scripts/pipelines/update_podcast_knowledge.py`
- Modify: `scripts/pipelines/update_vector_database.py`

- [ ] **Step 1: Export `PodcastVectorManager`**

Replace the full contents of `data/postgres/vector_managers/__init__.py`:

```python
"""
Vector managers package for PostgreSQL with pgvector.
"""

from data.postgres.vector_managers.base import BaseVectorManager
from data.postgres.vector_managers.youtube import (
    YoutubeVectorManager,
    YoutubeChannelVectorManager
)
from data.postgres.vector_managers.article import ArticleVectorManager
from data.postgres.vector_managers.podcast import PodcastVectorManager

__all__ = [
    'BaseVectorManager',
    'YoutubeVectorManager',
    'YoutubeChannelVectorManager',
    'ArticleVectorManager',
    'PodcastVectorManager',
]
```

- [ ] **Step 2: Create `scripts/pipelines/update_podcast_knowledge.py`**

```python
"""
Script to update podcast knowledge from RSS feeds via Whisper transcription.

Usage:
    python scripts/pipelines/update_podcast_knowledge.py
"""

import os
import sys
import yaml
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data.postgres.connection import PostgresConnection
from data.postgres.vector_managers import PodcastVectorManager
from langchain_openai import OpenAIEmbeddings
from logger import get_logger, setup_logging

load_dotenv(project_root / '.env')
setup_logging(debug=False)
logger = get_logger(__name__)

vector_config_path = project_root / 'config' / 'vector_config.yaml'
vector_config = {}
if vector_config_path.exists():
    with open(vector_config_path, 'r') as f:
        vector_config = yaml.safe_load(f) or {}
else:
    logger.warning(f"Vector config not found at {vector_config_path}")


def main():
    podcast_config = vector_config.get('podcast', {})
    feeds = podcast_config.get('feeds', [])
    if not feeds:
        logger.warning("No podcast feeds in config/vector_config.yaml under 'podcast.feeds'")
        return

    max_episodes_added = podcast_config.get('max_episodes_added', 5)
    chunk_size_seconds = podcast_config.get('chunk_size_seconds', 120)
    delay_between_episodes = podcast_config.get('delay_between_episodes', 2.0)

    connection = PostgresConnection()
    embeddings = OpenAIEmbeddings(api_key=os.environ.get('OPENAI_API_KEY'))
    manager = PodcastVectorManager(
        connection=connection,
        embeddings=embeddings,
        table_name=None,
        chunk_size_seconds=chunk_size_seconds,
    )

    season_start = "2025-07-01"
    season_end = "2026-06-30"

    logger.info("=" * 60)
    logger.info("Podcast RSS Pipeline")
    logger.info("=" * 60)

    total_added, total_chunks = 0, 0
    for feed in feeds:
        name = feed.get('name', 'Unknown')
        rss_url = feed.get('rss_url')
        if not rss_url:
            logger.warning(f"Skipping '{name}' — no rss_url configured")
            continue
        logger.info(f"\nFeed: {name}")
        results = manager.add_feed_to_vectorstore(
            rss_url=rss_url,
            feed_name=name,
            max_episodes_added=max_episodes_added,
            season_start=season_start,
            season_end=season_end,
            skip_existing=True,
            delay_between_episodes=delay_between_episodes,
        )
        total_added += results["episodes_added"]
        total_chunks += results["chunks_added"]

    logger.info(f"\n✅ Podcast pipeline complete — episodes: {total_added}, chunks: {total_chunks:,}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Register in master `update_vector_database.py`**

Add import after existing imports at line ~33:
```python
from scripts.pipelines.update_podcast_knowledge import main as update_podcast_knowledge
```

Add to the `pipelines` list in `main()`:
```python
pipelines = [
    ("Base Knowledge Update (knowledge_base.yaml)", update_base_knowledge),
    ("LockedOn Knowledge Update", update_lockedon_knowledge),
    ("Podcast RSS Knowledge Update", update_podcast_knowledge),
]
```

- [ ] **Step 4: Verify imports load cleanly**

```bash
cd /Users/matt/Code/ClutchAI && python -c "from scripts.pipelines.update_vector_database import main; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Run full test suite one final time**

```bash
cd /Users/matt/Code/ClutchAI && python -m pytest tests/test_podcast_vector_manager.py -v 2>&1 | tail -20
```

Expected: All 17 tests PASS

- [ ] **Step 6: Final commit**

```bash
git add data/postgres/vector_managers/__init__.py \
        scripts/pipelines/update_podcast_knowledge.py \
        scripts/pipelines/update_vector_database.py
git commit -m "feat: wire up podcast pipeline to master cron and package exports"
```

---

## Self-Review Checklist

- [x] **RSS fetch + date filtering** → `fetch_feed_episodes` with `season_start`/`season_end`
- [x] **Audio download** → `_download_audio` with streaming to temp file
- [x] **Optimized Whisper** → `language="en"`, domain `prompt` with NBA names/terms, `temperature=0`, `verbose_json` with segment timestamps
- [x] **25 MB size guard** → `_transcribe_audio` raises `ValueError` with helpful message
- [x] **Transcript cleaning** → `_clean_segments` calls `gpt-4o-mini` to drop ad/intro/outro segments; falls back to raw segments on any error
- [x] **Timestamps preserved** → LLM filters segment indices, remaining segments keep `start`/`end` metadata
- [x] **120s chunks** → default `chunk_size_seconds=120` in config and class
- [x] **Skip existing** → `add_feed_to_vectorstore` checks `get_existing_resource_ids()`
- [x] **Max episodes per run** → respected in `add_feed_to_vectorstore`
- [x] **Temp file cleanup on error** → `finally` block in `load_resource_content`
- [x] **Config-driven feeds** → `config/vector_config.yaml` `podcast.feeds` list
- [x] **Master cron integrated** → registered in `update_vector_database.py`
- [x] **No new env vars** → reuses `OPENAI_API_KEY`
- [x] **feedparser already in requirements** → confirmed
- [x] **17 unit tests** — all external I/O mocked (feedparser, openai, requests, filesystem)
