"""
Integration tests for Cloud SQL vectordb connection.

These tests require:
- DATABASE_URL environment variable set

Run with: pytest -m integration tests/test_vectordb_connection.py
"""

import os
import pytest
from sqlalchemy import text

from data.cloud_sql.connection import PostgresConnection
from data.cloud_sql.schema import setup_pgvector_extension, get_table_stats, get_default_table_name


pytestmark = pytest.mark.skipif(
    not os.environ.get('DATABASE_URL'),
    reason="DATABASE_URL not set. Integration tests require a live PostgreSQL connection."
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
    def test_schema_setup_and_stats(self, save_test_output):
        """Test pgvector schema setup and table statistics.
        
        Schema information saved to: tests/test_outputs/test_vectordb_connection/test_schema_setup_and_stats_schema.txt
        """
        # Get collection name from environment variable
        collection_name = get_default_table_name()
        
        # Setup pgvector extension (PGVector manages its own tables)
        success = setup_pgvector_extension(self.connection)
        assert success is True
        
        # Collect schema information
        schema_output = []
        engine = self.connection.get_engine()
        with engine.connect() as conn:
            # Get PGVector table schema (langchain_pg_embedding)
            result = conn.execute(text("""
                SELECT 
                    column_name,
                    data_type,
                    character_maximum_length,
                    is_nullable,
                    column_default
                FROM information_schema.columns
                WHERE table_name = 'langchain_pg_embedding'
                ORDER BY ordinal_position
            """))
            
            schema_output.append("="*80)
            schema_output.append(f"SCHEMA: langchain_pg_embedding table (PGVector)")
            schema_output.append("="*80)
            for row in result.fetchall():
                col_name, data_type, max_length, nullable, default = row
                type_str = f"{data_type}"
                if max_length:
                    type_str += f"({max_length})"
                nullable_str = "NULL" if nullable == "YES" else "NOT NULL"
                default_str = f" DEFAULT {default}" if default else ""
                schema_output.append(f"  {col_name:20} {type_str:30} {nullable_str}{default_str}")
            
            # Get indexes
            result = conn.execute(text("""
                SELECT 
                    indexname,
                    indexdef
                FROM pg_indexes
                WHERE tablename = 'langchain_pg_embedding'
            """))
            
            indexes = result.fetchall()
            if indexes:
                schema_output.append("\nINDEXES:")
                for idx_name, idx_def in indexes:
                    schema_output.append(f"  {idx_name}:")
                    schema_output.append(f"    {idx_def}")
            else:
                schema_output.append("\nINDEXES: None")
            
            schema_output.append("="*80)
        
        # Get stats
        stats = get_table_stats(self.connection, table_name=collection_name)
        
        # Add stats to schema output
        schema_output.append("\nTABLE STATISTICS:")
        schema_output.append("="*80)
        schema_output.append(f"Row Count: {stats.get('row_count', 0)}")
        schema_output.append(f"Unique Resources: {stats.get('unique_resources', 0)}")
        schema_output.append(f"Source Types: {stats.get('source_types', {})}")
        schema_output.append("="*80)
        
        # Save schema output to file
        schema_text = "\n".join(schema_output)
        save_test_output("schema", schema_text)
        
        # Verify stats
        assert 'error' not in stats
        assert 'row_count' in stats
        assert 'unique_resources' in stats
        assert 'source_types' in stats
        assert isinstance(stats['row_count'], int)
        assert isinstance(stats['unique_resources'], int)