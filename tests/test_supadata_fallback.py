"""
Tests for Supadata transcript fallback in YoutubeVectorManager.
"""
import os
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager(monkeypatch):
    """Return a YoutubeVectorManager with minimal mocked dependencies (no real DB)."""
    monkeypatch.setenv('DATABASE_URL', 'postgresql://user:pass@localhost:5432/testdb')
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-fake')

    with patch('data.postgres.vector_managers.base.PGVector'):
        from data.postgres.connection import PostgresConnection
        from data.postgres.vector_managers.youtube import YoutubeVectorManager
        from langchain_openai import OpenAIEmbeddings

        conn = PostgresConnection()
        embeddings = MagicMock(spec=OpenAIEmbeddings)
        return YoutubeVectorManager(
            connection=conn,
            embeddings=embeddings,
            table_name='test_table',
            chunk_size_seconds=30,
            max_retries=0,  # No retries so tests run fast
            retry_delay=0.0,
        )


SUPADATA_RESPONSE = {
    "videoId": "abc123",
    "lang": "en",
    "content": [
        {"text": "Hello everyone welcome back", "offset": 0, "duration": 3000},
        {"text": "to the show today we talk", "offset": 3000, "duration": 3000},
        {"text": "about fantasy basketball picks", "offset": 6000, "duration": 3000},
        # This segment starts a new chunk (offset 35s > chunk_size 30s)
        {"text": "second chunk begins here now", "offset": 35000, "duration": 3000},
        {"text": "more content in second chunk", "offset": 38000, "duration": 3000},
    ],
}


# ---------------------------------------------------------------------------
# Unit tests — no network calls
# ---------------------------------------------------------------------------

def test_supadata_chunks_segments_correctly(monkeypatch):
    """_fetch_transcript_supadata groups segments into chunk_size_seconds chunks."""
    manager = _make_manager(monkeypatch)
    monkeypatch.setenv('SUPADATA_API_KEY', 'fake-key')

    mock_response = MagicMock()
    mock_response.json.return_value = SUPADATA_RESPONSE
    mock_response.raise_for_status = MagicMock()

    with patch('requests.get', return_value=mock_response):
        docs = manager._fetch_transcript_supadata('abc123', chunk_size_seconds=30)

    assert len(docs) == 2
    assert 'Hello everyone welcome back' in docs[0].page_content
    assert 'second chunk begins here now' in docs[1].page_content
    assert docs[0].metadata['start_seconds'] == 0.0
    assert docs[1].metadata['start_seconds'] == 35.0


def test_supadata_raises_without_api_key(monkeypatch):
    """_fetch_transcript_supadata raises ValueError when SUPADATA_API_KEY is not set."""
    manager = _make_manager(monkeypatch)
    monkeypatch.delenv('SUPADATA_API_KEY', raising=False)

    with pytest.raises(ValueError, match="SUPADATA_API_KEY"):
        manager._fetch_transcript_supadata('abc123', chunk_size_seconds=30)


def test_supadata_raises_on_empty_transcript(monkeypatch):
    """_fetch_transcript_supadata raises when Supadata returns no content."""
    manager = _make_manager(monkeypatch)
    monkeypatch.setenv('SUPADATA_API_KEY', 'fake-key')

    mock_response = MagicMock()
    mock_response.json.return_value = {"videoId": "abc123", "content": []}
    mock_response.raise_for_status = MagicMock()

    with patch('requests.get', return_value=mock_response):
        with pytest.raises(ValueError, match="no transcript"):
            manager._fetch_transcript_supadata('abc123', chunk_size_seconds=30)


def test_ip_block_triggers_supadata_fallback(monkeypatch):
    """When YoutubeLoader raises an IP blocking error, Supadata fallback is used."""
    manager = _make_manager(monkeypatch)
    monkeypatch.setenv('SUPADATA_API_KEY', 'fake-key')

    mock_response = MagicMock()
    mock_response.json.return_value = SUPADATA_RESPONSE
    mock_response.raise_for_status = MagicMock()

    with patch('langchain_community.document_loaders.YoutubeLoader.from_youtube_url') as mock_loader, \
         patch('requests.get', return_value=mock_response):

        mock_loader.return_value.load.side_effect = Exception("IP blocking detected")

        docs = manager.load_resource_content(
            url='https://www.youtube.com/watch?v=abc123',
            resource_id='abc123',
        )

    assert len(docs) == 2


def test_non_ip_error_does_not_trigger_supadata(monkeypatch):
    """Non-IP-block errors (e.g. no transcript available) skip Supadata and raise."""
    manager = _make_manager(monkeypatch)
    monkeypatch.setenv('SUPADATA_API_KEY', 'fake-key')

    with patch('langchain_community.document_loaders.YoutubeLoader.from_youtube_url') as mock_loader, \
         patch('requests.get') as mock_requests:

        mock_loader.return_value.load.side_effect = Exception("No transcript available for this video")

        with pytest.raises(ValueError):
            manager.load_resource_content(
                url='https://www.youtube.com/watch?v=abc123',
                resource_id='abc123',
            )

        mock_requests.assert_not_called()


# ---------------------------------------------------------------------------
# Integration test — requires SUPADATA_API_KEY
# ---------------------------------------------------------------------------

pytestmark_integration = pytest.mark.skipif(
    not os.environ.get('SUPADATA_API_KEY'),
    reason="SUPADATA_API_KEY not set"
)


@pytestmark_integration
def test_supadata_real_api_call(monkeypatch):
    """Actually calls Supadata with a known LockedOn Fantasy Basketball video."""
    monkeypatch.setenv('DATABASE_URL', 'postgresql://user:pass@localhost:5432/testdb')
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-fake')

    from data.postgres.connection import PostgresConnection
    from data.postgres.vector_managers.youtube import YoutubeVectorManager
    from langchain_openai import OpenAIEmbeddings

    conn = PostgresConnection()
    embeddings = MagicMock(spec=OpenAIEmbeddings)
    manager = YoutubeVectorManager(
        connection=conn,
        embeddings=embeddings,
        table_name='test_table',
        chunk_size_seconds=30,
    )

    # A recent LockedOn Fantasy Basketball episode
    video_id = 'dQw4w9WgXcQ'  # Replace with a real LockedOn video ID before running
    docs = manager._fetch_transcript_supadata(video_id, chunk_size_seconds=30)

    assert len(docs) > 0
    assert all(isinstance(d, Document) for d in docs)
    assert all('start_seconds' in d.metadata for d in docs)
    assert all(len(d.page_content) > 0 for d in docs)
