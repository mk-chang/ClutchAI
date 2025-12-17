"""
Cloud SQL PostgreSQL Schema Management using SQLAlchemy

Handles pgvector extension setup and vector table creation.
"""

import os
from typing import Optional
from sqlalchemy import text
from data.cloud_sql.connection import PostgresConnection
from logger import get_logger

logger = get_logger(__name__)


def get_default_table_name() -> str:
    """
    Get the default vector table name from environment variables.
    
    Returns:
        Vector table name to use
        
    Raises:
        ValueError: If CLOUDSQL_VECTOR_TABLE is not set
    """
    # Check for explicit table name
    table_name = os.environ.get('CLOUDSQL_VECTOR_TABLE')
    if table_name:
        return table_name
    
    # Require explicit table name - no fallback
    raise ValueError(
        "Vector table name is required. "
        "Set CLOUDSQL_VECTOR_TABLE environment variable to specify the table name."
    )


def get_app_table_name() -> str:
    """
    Get the app table name from environment variables.
    
    Priority:
    1. CLOUDSQL_APP_TABLE (if set)
    2. "clutchai_app" (default fallback)
    
    Returns:
        App table name to use
    """
    # Check for explicit table name
    table_name = os.environ.get('CLOUDSQL_APP_TABLE')
    if table_name:
        return table_name
    
    # Default fallback
    return "clutchai_app"


def setup_pgvector_extension(connection: PostgresConnection) -> bool:
    """
    Set up pgvector extension only (PGVector manages its own tables).
    
    Args:
        connection: PostgresConnection instance
        
    Returns:
        True if successful, False otherwise
    """
    try:
        engine = connection.get_engine()
        
        with engine.connect() as conn:
            # Enable pgvector extension
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
        
        logger.info("Set up pgvector extension")
        return True
    except Exception as e:
        logger.error(f"Failed to setup pgvector extension: {e}")
        logger.warning("Note: Make sure pgvector extension is available in your PostgreSQL instance.")
        logger.warning("For Cloud SQL, enable the 'pgvector' flag when creating the instance.")
        return False


def setup_schema(connection: PostgresConnection, vector_size: int = 1536, table_name: Optional[str] = None) -> bool:
    """
    Set up pgvector extension and vector table.
    
    DEPRECATED: This function creates a custom table. Use setup_pgvector_extension() instead
    and let PGVector manage its own tables via collection_name.
    
    Args:
        connection: PostgresConnection instance
        vector_size: Dimension of embedding vectors (default: 1536 for OpenAI)
        table_name: Name of the vector table (defaults to env var CLOUDSQL_VECTOR_TABLE)
        
    Returns:
        True if successful, False otherwise
    """
    if table_name is None:
        table_name = get_default_table_name()
    
    try:
        engine = connection.get_engine()
        
        with engine.connect() as conn:
            # Enable pgvector extension
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
            
            # Create vector table if it doesn't exist
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id SERIAL PRIMARY KEY,
                    embedding vector({vector_size}),
                    document TEXT,
                    metadata JSONB
                )
            """))
            conn.commit()
        
        logger.info(f"Set up pgvector extension and vector table '{table_name}'")
        return True
    except Exception as e:
        logger.error(f"Failed to setup schema: {e}")
        logger.warning("Note: Make sure pgvector extension is available in your PostgreSQL instance.")
        logger.warning("For Cloud SQL, enable the 'pgvector' flag when creating the instance.")
        return False


def create_vector_table(
    connection: PostgresConnection,
    table_name: Optional[str] = None,
    vector_dimension: int = 1536,
    drop_existing: bool = False
) -> bool:
    """
    Create vector table with pgvector support.
    
    Args:
        connection: PostgresConnection instance
        table_name: Name of the vector table (defaults to env var CLOUDSQL_VECTOR_TABLE)
        vector_dimension: Dimension of embedding vectors (default: 1536 for OpenAI)
        drop_existing: If True, drop existing table before creating
        
    Returns:
        True if successful, False otherwise
    """
    if table_name is None:
        table_name = get_default_table_name()
    
    try:
        engine = connection.get_engine()
        
        with engine.connect() as conn:
            # Drop existing table if requested
            if drop_existing:
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
                conn.commit()
            
            # Create vector table
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id SERIAL PRIMARY KEY,
                    embedding vector({vector_dimension}),
                    document TEXT,
                    metadata JSONB
                )
            """))
            conn.commit()
        
        logger.info(f"Created/verified vector table {table_name} with pgvector support")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create vector table: {e}")
        return False


def get_table_stats(connection: PostgresConnection, table_name: Optional[str] = None) -> dict:
    """
    Get statistics about the PGVector collection.
    
    Queries PGVector's langchain_pg_embedding table for the specified collection.
    
    Args:
        connection: PostgresConnection instance
        table_name: Collection name (defaults to env var CLOUDSQL_VECTOR_TABLE)
        
    Returns:
        Dictionary with collection statistics
    """
    if table_name is None:
        table_name = get_default_table_name()
    
    try:
        engine = connection.get_engine()
        
        with engine.connect() as conn:
            # First, get the collection UUID
            result = conn.execute(text("""
                SELECT uuid FROM langchain_pg_collection 
                WHERE name = :collection_name
            """), {"collection_name": table_name})
            collection_row = result.fetchone()
            
            if not collection_row:
                return {
                    'collection_name': table_name,
                    'row_count': 0,
                    'unique_resources': 0,
                    'source_types': {},
                    'error': 'Collection not found'
                }
            
            collection_uuid = collection_row[0]
            
            # Get row count from langchain_pg_embedding
            result = conn.execute(text("""
                SELECT COUNT(*) FROM langchain_pg_embedding 
                WHERE collection_id = :collection_uuid
            """), {"collection_uuid": collection_uuid})
            row_count = result.scalar() or 0
            
            # Get unique resources from metadata
            result = conn.execute(text("""
                SELECT COUNT(DISTINCT (cmetadata->>'resource_id')) 
                FROM langchain_pg_embedding 
                WHERE collection_id = :collection_uuid
                AND cmetadata->>'resource_id' IS NOT NULL
            """), {"collection_uuid": collection_uuid})
            unique_resources = result.scalar() or 0
            
            # Get source types distribution
            result = conn.execute(text("""
                SELECT cmetadata->>'source_type' as source_type, COUNT(*) as count
                FROM langchain_pg_embedding
                WHERE collection_id = :collection_uuid
                AND cmetadata->>'source_type' IS NOT NULL
                GROUP BY cmetadata->>'source_type'
            """), {"collection_uuid": collection_uuid})
            source_types = {row[0]: row[1] for row in result.fetchall()}
        
        return {
            'collection_name': table_name,
            'row_count': row_count,
            'unique_resources': unique_resources,
            'source_types': source_types
        }
    except Exception as e:
        return {'error': str(e)}


def create_ivfflat_index(
    connection: PostgresConnection,
    table_name: Optional[str] = None,
    column_name: str = "embedding",
    lists: Optional[int] = None,
    drop_existing: bool = False
) -> bool:
    """
    Create an IVFFlat index on PGVector's embedding column for efficient similarity search.
    
    IVFFlat is an approximate nearest neighbor index that significantly speeds up
    vector similarity searches. The index should be created after you have some
    data in the collection (at least a few rows, preferably 100+ for good performance).
    
    This function creates an index on PGVector's langchain_pg_embedding table.
    
    Args:
        connection: PostgresConnection instance
        table_name: Collection name (defaults to env var CLOUDSQL_VECTOR_TABLE)
        column_name: Name of the vector column in langchain_pg_embedding (default: "embedding")
        lists: Number of clusters/lists for the index. If None, auto-calculated
               based on row count (recommended: rows/1000, min 10).
               For best performance, provide a value based on your dataset size.
        drop_existing: If True, drop existing index before creating
        
    Returns:
        True if successful, False otherwise
    """
    if table_name is None:
        table_name = get_default_table_name()
    
    try:
        engine = connection.get_engine()
        # Use collection name in index name for clarity
        index_name = f"langchain_pg_embedding_{table_name}_{column_name}_ivfflat_idx"
        
        with engine.connect() as conn:
            # Get row count from langchain_pg_embedding for the collection
            # First get collection UUID
            result = conn.execute(text("""
                SELECT uuid FROM langchain_pg_collection 
                WHERE name = :collection_name
            """), {"collection_name": table_name})
            collection_row = result.fetchone()
            
            if not collection_row:
                logger.warning(f"Collection '{table_name}' not found. IVFFlat index requires an existing collection.")
                return False
            
            collection_uuid = collection_row[0]
            
            # Get row count for this collection
            result = conn.execute(text("""
                SELECT COUNT(*) FROM langchain_pg_embedding 
                WHERE collection_id = :collection_uuid
            """), {"collection_uuid": collection_uuid})
            row_count = result.scalar() or 0
            
            if row_count == 0:
                logger.warning(f"Collection '{table_name}' is empty. IVFFlat index requires data.")
                logger.warning("Consider adding some vectors before creating the index.")
                return False
            
            # Calculate lists if not provided
            if lists is None:
                # Recommended: rows/1000 for up to 1M rows
                # Minimum 10, round to nearest integer
                calculated_lists = max(10, int(row_count / 1000))
                lists = calculated_lists
                logger.info(f"Auto-calculated lists parameter: {lists} (based on {row_count} rows)")
            
            # Drop existing index if requested
            if drop_existing:
                conn.execute(text(f"DROP INDEX IF EXISTS {index_name}"))
                conn.commit()
            
            # Create IVFFlat index on langchain_pg_embedding table
            # Using vector_cosine_ops for cosine similarity (most common for embeddings)
            # Alternatives: vector_l2_ops (L2 distance), vector_ip_ops (inner product)
            # Note: Index is created on all embeddings, but you can filter by collection_id in queries
            conn.execute(text(f"""
                CREATE INDEX {index_name}
                ON langchain_pg_embedding
                USING ivfflat ({column_name} vector_cosine_ops)
                WITH (lists = {lists})
            """))
            conn.commit()
        
        logger.info(f"Created IVFFlat index '{index_name}' on langchain_pg_embedding.{column_name} for collection '{table_name}' with lists={lists}")
        logger.info(f"Index will speed up similarity searches on {row_count} vectors")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create IVFFlat index: {e}")
        logger.warning("Note: IVFFlat index requires:")
        logger.warning("  - Collection to exist with at least some data")
        logger.warning("  - pgvector extension to be enabled")
        logger.warning("  - Vector column to exist in langchain_pg_embedding")
        return False
