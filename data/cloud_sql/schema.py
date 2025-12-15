"""
Cloud SQL PostgreSQL Schema Management using SQLAlchemy

Handles pgvector extension setup and vector table creation.
"""

from typing import Optional
from sqlalchemy import text
from data.cloud_sql.connection import PostgresConnection


def setup_schema(connection: PostgresConnection, vector_size: int = 1536) -> bool:
    """
    Set up pgvector extension and vector table.
    
    Args:
        connection: PostgresConnection instance
        vector_size: Dimension of embedding vectors (default: 1536 for OpenAI)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        engine = connection.get_engine()
        
        with engine.connect() as conn:
            # Enable pgvector extension
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
            
            # Create vector table if it doesn't exist
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id SERIAL PRIMARY KEY,
                    embedding vector({vector_size}),
                    document TEXT,
                    metadata JSONB
                )
            """))
            conn.commit()
        
        print("✓ Set up pgvector extension and vector table")
        return True
    except Exception as e:
        print(f"✗ Failed to setup schema: {e}")
        print("  Note: Make sure pgvector extension is available in your PostgreSQL instance.")
        print("  For Cloud SQL, enable the 'pgvector' flag when creating the instance.")
        return False


def create_vector_table(
    connection: PostgresConnection,
    table_name: str = "embeddings",
    vector_dimension: int = 1536,
    drop_existing: bool = False
) -> bool:
    """
    Create vector table with pgvector support.
    
    Args:
        connection: PostgresConnection instance
        table_name: Name of the vector table
        vector_dimension: Dimension of embedding vectors (default: 1536 for OpenAI)
        drop_existing: If True, drop existing table before creating
        
    Returns:
        True if successful, False otherwise
    """
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
        
        print(f"✓ Created/verified vector table {table_name} with pgvector support")
        return True
        
    except Exception as e:
        print(f"✗ Failed to create vector table: {e}")
        return False


def get_table_stats(connection: PostgresConnection, table_name: str = "embeddings") -> dict:
    """
    Get statistics about the vector table.
    
    Args:
        connection: PostgresConnection instance
        table_name: Name of the vector table
        
    Returns:
        Dictionary with table statistics
    """
    try:
        engine = connection.get_engine()
        
        with engine.connect() as conn:
            # Get row count
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            row_count = result.scalar()
            
            # Get unique resources
            result = conn.execute(text(f"""
                SELECT COUNT(DISTINCT metadata->>'resource_id') 
                FROM {table_name} 
                WHERE metadata->>'resource_id' IS NOT NULL
            """))
            unique_resources = result.scalar() or 0
            
            # Get source types distribution
            result = conn.execute(text(f"""
                SELECT metadata->>'source_type' as source_type, COUNT(*) as count
                FROM {table_name}
                WHERE metadata->>'source_type' IS NOT NULL
                GROUP BY metadata->>'source_type'
            """))
            source_types = {row[0]: row[1] for row in result.fetchall()}
        
        return {
            'table_name': table_name,
            'row_count': row_count,
            'unique_resources': unique_resources,
            'source_types': source_types
        }
    except Exception as e:
        return {'error': str(e)}


def create_ivfflat_index(
    connection: PostgresConnection,
    table_name: str = "embeddings",
    column_name: str = "embedding",
    lists: Optional[int] = None,
    drop_existing: bool = False
) -> bool:
    """
    Create an IVFFlat index on the vector column for efficient similarity search.
    
    IVFFlat is an approximate nearest neighbor index that significantly speeds up
    vector similarity searches. The index should be created after you have some
    data in the table (at least a few rows, preferably 100+ for good performance).
    
    Args:
        connection: PostgresConnection instance
        table_name: Name of the vector table
        column_name: Name of the vector column (default: "embedding")
        lists: Number of clusters/lists for the index. If None, auto-calculated
               based on row count (recommended: rows/1000, min 10).
               For best performance, provide a value based on your dataset size.
        drop_existing: If True, drop existing index before creating
        
    Returns:
        True if successful, False otherwise
    """
    try:
        engine = connection.get_engine()
        index_name = f"{table_name}_{column_name}_ivfflat_idx"
        
        with engine.connect() as conn:
            # Check if table exists and get row count
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            row_count = result.scalar()
            
            if row_count == 0:
                print(f"⚠ Warning: Table {table_name} is empty. IVFFlat index requires data.")
                print("  Consider adding some vectors before creating the index.")
                return False
            
            # Calculate lists if not provided
            if lists is None:
                # Recommended: rows/1000 for up to 1M rows
                # Minimum 10, round to nearest integer
                calculated_lists = max(10, int(row_count / 1000))
                lists = calculated_lists
                print(f"  Auto-calculated lists parameter: {lists} (based on {row_count} rows)")
            
            # Drop existing index if requested
            if drop_existing:
                conn.execute(text(f"DROP INDEX IF EXISTS {index_name}"))
                conn.commit()
            
            # Create IVFFlat index
            # Using vector_cosine_ops for cosine similarity (most common for embeddings)
            # Alternatives: vector_l2_ops (L2 distance), vector_ip_ops (inner product)
            conn.execute(text(f"""
                CREATE INDEX {index_name}
                ON {table_name}
                USING ivfflat ({column_name} vector_cosine_ops)
                WITH (lists = {lists})
            """))
            conn.commit()
        
        print(f"✓ Created IVFFlat index '{index_name}' on {table_name}.{column_name} with lists={lists}")
        print(f"  Index will speed up similarity searches on {row_count} vectors")
        return True
        
    except Exception as e:
        print(f"✗ Failed to create IVFFlat index: {e}")
        print("  Note: IVFFlat index requires:")
        print("    - Table to exist with at least some data")
        print("    - pgvector extension to be enabled")
        print("    - Vector column to exist")
        return False
