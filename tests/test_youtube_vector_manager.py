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
