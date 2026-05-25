# YouTube Transcript Cleaning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After YouTube transcript chunks are fetched (via YoutubeLoader or Supadata), filter out chunks containing ads, sponsor reads, intros, outros, and social media plugs using `gpt-4o-mini` before storing in pgvector.

**Architecture:** Add `_clean_documents(docs)` to `BaseVectorManager` — reusable by both YouTube and future podcast ingestion. Call it in `YoutubeVectorManager.load_resource_content()` immediately after docs are fetched, before metadata enhancement. Always-on, gracefully falls back to raw docs on any LLM error.

**Tech Stack:** `gpt-4o-mini` via `openai` (same `OPENAI_API_KEY`), LangChain `Document`. No new dependencies.

---

## File Map

| Action | File | Change |
|--------|------|--------|
| Modify | `data/postgres/vector_managers/base.py` | Add `_clean_documents()` method |
| Modify | `data/postgres/vector_managers/youtube.py` | Call `_clean_documents()` in `load_resource_content()` |
| Create | `tests/test_youtube_vector_manager.py` | Tests for `_clean_documents` and the integrated pipeline |

---

## Task 1: Write failing tests for `_clean_documents` → implement → pass

**Files:**
- Create: `tests/test_youtube_vector_manager.py`
- Modify: `data/postgres/vector_managers/base.py`

- [ ] **Step 1: Write the tests**

Create `tests/test_youtube_vector_manager.py`:

```python
"""Tests for YoutubeVectorManager transcript cleaning."""
import json
import pytest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from data.postgres.vector_managers.youtube import YoutubeVectorManager


@pytest.fixture
def manager():
    mock_conn = MagicMock()
    mock_conn.get_engine.return_value = MagicMock()
    with patch('data.postgres.vector_managers.base.setup_pgvector_extension'), \
         patch('data.postgres.vector_managers.base.PGVector'):
        return YoutubeVectorManager(
            connection=mock_conn,
            embeddings=MagicMock(),
            table_name='test_table',
        )


def _make_docs(texts):
    return [Document(page_content=t, metadata={"source": f"http://yt.com?t={i}s"})
            for i, t in enumerate(texts)]


def _mock_openai(indices_to_remove):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps({"remove": indices_to_remove})
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


class TestCleanDocuments:

    def test_removes_ad_chunks(self, manager):
        docs = _make_docs([
            "Welcome to the show! I'm your host.",
            "Today's episode is brought to you by DraftKings.",
            "Use promo code LOCKED for 20% off your first deposit.",
            "Alright, let's talk about Nikola Jokic.",
            "His rebounding numbers this week have been elite.",
            "Follow us on Twitter and leave a five-star review.",
        ])
        with patch('data.postgres.vector_managers.base.OpenAI',
                   return_value=_mock_openai([0, 1, 2, 5])):
            clean = manager._clean_documents(docs)
        texts = [d.page_content for d in clean]
        assert "Alright, let's talk about Nikola Jokic." in texts
        assert "His rebounding numbers this week have been elite." in texts
        assert "Today's episode is brought to you by DraftKings." not in texts
        assert "Use promo code LOCKED for 20% off your first deposit." not in texts

    def test_returns_all_docs_when_nothing_to_remove(self, manager):
        docs = _make_docs([
            "Jokic drops 40 points and 15 rebounds.",
            "Should you trade for him on waivers?",
        ])
        with patch('data.postgres.vector_managers.base.OpenAI',
                   return_value=_mock_openai([])):
            clean = manager._clean_documents(docs)
        assert len(clean) == 2

    def test_falls_back_to_raw_docs_on_llm_error(self, manager):
        docs = _make_docs(["Content A", "Content B"])
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API error")
        with patch('data.postgres.vector_managers.base.OpenAI', return_value=mock_client):
            clean = manager._clean_documents(docs)
        assert len(clean) == 2

    def test_falls_back_on_invalid_json(self, manager):
        docs = _make_docs(["Content A", "Content B"])
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "not valid json at all"
        mock_client.chat.completions.create.return_value = mock_response
        with patch('data.postgres.vector_managers.base.OpenAI', return_value=mock_client):
            clean = manager._clean_documents(docs)
        assert len(clean) == 2

    def test_returns_empty_list_for_empty_input(self, manager):
        with patch('data.postgres.vector_managers.base.OpenAI'):
            clean = manager._clean_documents([])
        assert clean == []

    def test_preserves_document_metadata(self, manager):
        docs = [Document(page_content="Good analysis here.",
                         metadata={"source": "http://yt.com?t=120s",
                                   "resource_id": "abc123",
                                   "title": "Episode 1"})]
        with patch('data.postgres.vector_managers.base.OpenAI',
                   return_value=_mock_openai([])):
            clean = manager._clean_documents(docs)
        assert clean[0].metadata["resource_id"] == "abc123"
        assert clean[0].metadata["title"] == "Episode 1"
```

- [ ] **Step 2: Run to confirm they fail**

```bash
cd /Users/matt/Code/ClutchAI && python -m pytest tests/test_youtube_vector_manager.py::TestCleanDocuments -v 2>&1 | tail -20
```

Expected: FAIL — `AttributeError: 'YoutubeVectorManager' object has no attribute '_clean_documents'`

- [ ] **Step 3: Implement `_clean_documents` in `BaseVectorManager`**

Add `import json` and `import os` to the existing imports at the top of `data/postgres/vector_managers/base.py` (both are likely already present — check first).

Then add this method to `BaseVectorManager` after `_add_documents_to_vectorstore` (~line 250):

```python
def _clean_documents(self, docs: List[Document]) -> List[Document]:
    if not docs:
        return docs

    from openai import OpenAI
    import json as _json

    numbered = "\n".join(
        f"[{i}] {doc.page_content.strip()}"
        for i, doc in enumerate(docs)
    )
    prompt = (
        "You are filtering a fantasy basketball video/podcast transcript. "
        "Identify chunk indices that contain: advertisements, sponsor reads, "
        "promo codes, show introductions, outros, social media promotions, "
        "calls to subscribe/follow/review, or any filler unrelated to basketball analysis. "
        "Return ONLY a JSON object with key 'remove' containing a list of integer indices. "
        "Do NOT remove any basketball analysis, player discussion, trade talk, "
        "injury news, or statistical analysis. "
        "If nothing should be removed, return {\"remove\": []}.\n\n"
        f"Chunks:\n{numbered}"
    )

    try:
        client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        result = _json.loads(response.choices[0].message.content)
        indices_to_remove = set(result.get("remove", []))
        kept = [doc for i, doc in enumerate(docs) if i not in indices_to_remove]
        removed = len(docs) - len(kept)
        if removed:
            logger.info(f"  Cleaned transcript: removed {removed} non-content chunks "
                        f"({len(kept)}/{len(docs)} kept)")
        return kept
    except Exception as e:
        logger.warning(f"  Transcript cleaning failed ({e}), using raw chunks")
        return docs
```

- [ ] **Step 4: Run to confirm tests pass**

```bash
cd /Users/matt/Code/ClutchAI && python -m pytest tests/test_youtube_vector_manager.py::TestCleanDocuments -v 2>&1 | tail -20
```

Expected: All 6 PASS

- [ ] **Step 5: Commit**

```bash
git add data/postgres/vector_managers/base.py tests/test_youtube_vector_manager.py
git commit -m "feat: add _clean_documents to BaseVectorManager with gpt-4o-mini filtering"
```

---

## Task 2: Call `_clean_documents` in `YoutubeVectorManager.load_resource_content`

**Files:**
- Modify: `data/postgres/vector_managers/youtube.py`
- Modify: `tests/test_youtube_vector_manager.py`

- [ ] **Step 1: Write failing integration test**

Append to `tests/test_youtube_vector_manager.py`:

```python
class TestLoadResourceContentCleaning:

    def test_clean_documents_called_after_fetch(self, manager):
        """_clean_documents must be called before metadata enhancement."""
        raw_docs = [
            Document(page_content="Use promo code LOCKED!", metadata={"source": "http://yt.com?t=0s"}),
            Document(page_content="Jokic analysis today.", metadata={"source": "http://yt.com?t=30s"}),
        ]
        clean_docs = [raw_docs[1]]  # ad chunk removed

        with patch('data.postgres.vector_managers.youtube.YoutubeLoader') as mock_loader_cls, \
             patch.object(manager, '_clean_documents', return_value=clean_docs) as mock_clean, \
             patch.object(manager, '_fetch_video_metadata_from_api', return_value=None):
            mock_loader_cls.from_youtube_url.return_value.load.return_value = raw_docs
            docs = manager.load_resource_content(
                url="https://www.youtube.com/watch?v=test123",
                source_type="youtube",
                title="Test Video",
                resource_id="test123",
            )

        mock_clean.assert_called_once_with(raw_docs)
        assert len(docs) == 1
        assert "Jokic analysis" in docs[0].page_content

    def test_metadata_applied_to_cleaned_docs_only(self, manager):
        """Metadata enhancement runs on the post-clean doc list."""
        raw_docs = [
            Document(page_content="Ad chunk.", metadata={"source": "http://yt.com?t=0s"}),
            Document(page_content="Good content.", metadata={"source": "http://yt.com?t=30s"}),
        ]
        with patch('data.postgres.vector_managers.youtube.YoutubeLoader') as mock_loader_cls, \
             patch.object(manager, '_clean_documents', return_value=[raw_docs[1]]), \
             patch.object(manager, '_fetch_video_metadata_from_api', return_value=None):
            mock_loader_cls.from_youtube_url.return_value.load.return_value = raw_docs
            docs = manager.load_resource_content(
                url="https://www.youtube.com/watch?v=test123",
                source_type="youtube",
                title="My Video",
                resource_id="test123",
            )

        assert len(docs) == 1
        assert docs[0].metadata.get("title") == "My Video"
        assert docs[0].metadata.get("source_type") == "youtube"
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
cd /Users/matt/Code/ClutchAI && python -m pytest tests/test_youtube_vector_manager.py::TestLoadResourceContentCleaning -v 2>&1 | tail -20
```

Expected: FAIL — `mock_clean.assert_called_once_with(raw_docs)` fails (cleaning not wired in yet)

- [ ] **Step 3: Add the `_clean_documents` call in `load_resource_content`**

In `data/postgres/vector_managers/youtube.py`, find the block that ends with:

```python
        if docs is None:
            raise ValueError(f"Failed to load transcript from {url}: {last_exception}") from last_exception

        # Auto-extract resource_id if not provided
        if not resource_id:
```

Insert the cleaning call between those two blocks:

```python
        if docs is None:
            raise ValueError(f"Failed to load transcript from {url}: {last_exception}") from last_exception

        # Clean transcript: remove ads, intros, outros, social plugs
        docs = self._clean_documents(docs)

        # Auto-extract resource_id if not provided
        if not resource_id:
```

- [ ] **Step 4: Run all tests to confirm everything passes**

```bash
cd /Users/matt/Code/ClutchAI && python -m pytest tests/test_youtube_vector_manager.py -v 2>&1 | tail -25
```

Expected: All 8 tests PASS

- [ ] **Step 5: Run the existing supadata test to confirm nothing is broken**

```bash
cd /Users/matt/Code/ClutchAI && python -m pytest tests/test_supadata_fallback.py -v 2>&1 | tail -20
```

Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add data/postgres/vector_managers/youtube.py tests/test_youtube_vector_manager.py
git commit -m "feat: clean YouTube transcript chunks before vectorstore ingestion"
```

---

## Self-Review Checklist

- [x] **`_clean_documents` on `BaseVectorManager`** — reusable by future podcast manager
- [x] **Always-on** — no config toggle, runs for every video
- [x] **Graceful fallback** — LLM error or bad JSON → return raw docs, log warning, continue
- [x] **Empty input guard** — returns immediately for empty doc list (no API call)
- [x] **Metadata preserved** — cleaning happens before metadata enhancement; kept docs get full metadata
- [x] **Correct insertion point** — after YoutubeLoader/Supadata, before metadata loop (~line 339 in youtube.py)
- [x] **8 tests** — 6 unit tests for `_clean_documents`, 2 integration tests for the wired-up pipeline
- [x] **No new dependencies** — `openai` already in requirements
