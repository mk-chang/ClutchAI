"""
Integration tests for RAG (Retrieval-Augmented Generation) functionality.

These tests require:
- OPENAI_API_KEY environment variable set
- GOOGLE_CLOUD_PROJECT environment variable set
- CLOUDSQL_REGION environment variable set
- CLOUDSQL_INSTANCE environment variable set
- CLOUDSQL_DATABASE environment variable set
- CLOUDSQL_USER environment variable set
- Vector database with some documents indexed

Run with: pytest -m integration tests/test_rag_integration.py
"""

import os
import pytest

from agents.rag.rag_manager import RAGManager


# Skip all integration tests if required environment variables are not set
pytestmark = pytest.mark.skipif(
    not all([
        os.environ.get('OPENAI_API_KEY'),
        os.environ.get('GOOGLE_CLOUD_PROJECT'),
        os.environ.get('CLOUDSQL_REGION'),
        os.environ.get('CLOUDSQL_INSTANCE'),
        os.environ.get('CLOUDSQL_DATABASE'),
        os.environ.get('CLOUDSQL_USER'),
    ]),
    reason="Required environment variables not set. Integration tests require OPENAI_API_KEY, GOOGLE_CLOUD_PROJECT, CLOUDSQL_REGION, CLOUDSQL_INSTANCE, CLOUDSQL_DATABASE, and CLOUDSQL_USER."
)


@pytest.mark.integration
class TestRAGIntegration:
    """Integration tests for RAG functionality."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup RAG manager for each test."""
        self.rag_manager = RAGManager()
        yield
    
    @pytest.mark.integration
    def test_rag_manager_initialization(self):
        """Test that RAG manager initializes correctly."""
        assert self.rag_manager is not None
        assert self.rag_manager.embeddings is not None
        assert self.rag_manager.table_name is not None
    
    @pytest.mark.integration
    def test_rag_retrieve(self, save_test_output):
        """Test RAG retrieval with a simple query."""
        query = "fantasy basketball"
        
        # Retrieve documents
        docs = self.rag_manager.retrieve(query, k=3)
        
        # Save results
        output_parts = [f"Query: {query}", f"Retrieved {len(docs)} documents", "="*80]
        for i, doc in enumerate(docs, 1):
            output_parts.append(f"\nDocument {i}:")
            output_parts.append(f"Title: {doc.metadata.get('title', 'N/A')}")
            output_parts.append(f"Source: {doc.metadata.get('url', doc.metadata.get('source', 'N/A'))}")
            output_parts.append(f"Content preview: {doc.page_content[:200]}...")
        
        output = "\n".join(output_parts)
        save_test_output("retrieve", output)
        
        # Basic assertions
        assert isinstance(docs, list)
        # Note: If vectorstore is empty, docs will be empty, which is fine for testing
    
    @pytest.mark.integration
    def test_rag_similarity_search(self, save_test_output):
        """Test RAG similarity search."""
        query = "player analysis"
        
        # Perform similarity search
        docs = self.rag_manager.similarity_search(query, k=2)
        
        # Save results
        output_parts = [f"Query: {query}", f"Found {len(docs)} documents", "="*80]
        for i, doc in enumerate(docs, 1):
            output_parts.append(f"\nDocument {i}:")
            output_parts.append(f"Title: {doc.metadata.get('title', 'N/A')}")
            output_parts.append(f"Source Type: {doc.metadata.get('source_type', 'N/A')}")
            output_parts.append(f"Content: {doc.page_content[:150]}...")
        
        output = "\n".join(output_parts)
        save_test_output("similarity_search", output)
        
        # Basic assertions
        assert isinstance(docs, list)
    
    @pytest.mark.integration
    def test_rag_get_retriever(self):
        """Test getting a retriever from RAG manager."""
        retriever = self.rag_manager.get_retriever(k=2)
        assert retriever is not None
        
        # Test invoking the retriever
        query = "basketball"
        docs = retriever.invoke(query)
        assert isinstance(docs, list)

