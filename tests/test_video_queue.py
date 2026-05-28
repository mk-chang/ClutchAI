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


def test_get_pending_includes_failed_under_max_attempts(mock_connection):
    mock_pg, mock_conn = mock_connection
    mock_row = MagicMock()
    mock_row._mapping = {
        "video_id": "xyz789",
        "title": "Failed Episode",
        "url": "https://www.youtube.com/watch?v=xyz789",
        "publish_date": None,
    }
    mock_conn.execute.return_value.fetchall.return_value = [mock_row]
    manager = VideoQueueManager(mock_pg)
    result = manager.get_pending(limit=5, max_attempts=3)
    assert len(result) == 1
    assert result[0]["video_id"] == "xyz789"


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
