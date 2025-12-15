"""
Integration tests for Cloud SQL vectordb connection.

These tests require:
- GOOGLE_CLOUD_PROJECT environment variable set
- CLOUDSQL_REGION environment variable set
- CLOUDSQL_INSTANCE environment variable set
- CLOUDSQL_VECTOR_DATABASE environment variable set
- CLOUDSQL_USER environment variable set
- Cloud SQL instance accessible and authorized

Run with: pytest -m integration tests/test_vectordb_connection.py
"""

import os
import pytest
from sqlalchemy import text

from data.cloud_sql.connection import PostgresConnection
from data.cloud_sql.schema import setup_schema, get_table_stats


# Skip all integration tests if Cloud SQL credentials are not set
pytestmark = pytest.mark.skipif(
    not all([
        os.environ.get('GOOGLE_CLOUD_PROJECT'),
        os.environ.get('CLOUDSQL_REGION'),
        os.environ.get('CLOUDSQL_INSTANCE'),
        os.environ.get('CLOUDSQL_VECTOR_DATABASE'),
        os.environ.get('CLOUDSQL_USER'),
    ]),
    reason="Cloud SQL environment variables not set. Integration tests require GOOGLE_CLOUD_PROJECT, CLOUDSQL_REGION, CLOUDSQL_INSTANCE, CLOUDSQL_VECTOR_DATABASE, and CLOUDSQL_USER."
)


@pytest.mark.integration
class TestVectordbConnection:
    """Integration tests for Cloud SQL vectordb connection."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup connection for each test."""
        self.connection = PostgresConnection()
        yield
        self.connection.close()
    
    @pytest.mark.integration
    def test_connection_and_version(self):
        """Test Cloud SQL connection and PostgreSQL version query."""
        engine = self.connection.get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            assert version is not None
            assert "PostgreSQL" in version
    
    @pytest.mark.integration
    def test_schema_setup_and_stats(self):
        """Test pgvector schema setup and table statistics."""
        # Setup schema
        success = setup_schema(self.connection)
        assert success is True
        
        # Get stats
        stats = get_table_stats(self.connection)
        assert 'error' not in stats
        assert 'row_count' in stats
        assert 'unique_resources' in stats
        assert 'source_types' in stats
        assert isinstance(stats['row_count'], int)
        assert isinstance(stats['unique_resources'], int)
