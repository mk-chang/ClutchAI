"""
Cloud SQL PostgreSQL Connection and Schema Management
"""

from data.cloud_sql.connection import PostgresConnection
from data.cloud_sql.schema import setup_pgvector_extension, setup_schema, create_vector_table
from data.cloud_sql.vector_managers import (
    BaseVectorManager,
    YoutubeVectorManager,
    YoutubeChannelVectorManager,
    ArticleVectorManager
)

__all__ = [
    'PostgresConnection',
    'setup_pgvector_extension',
    'setup_schema',  # Deprecated, use setup_pgvector_extension
    'create_vector_table',  # Deprecated, PGVector manages its own tables
    'BaseVectorManager',
    'YoutubeVectorManager',
    'YoutubeChannelVectorManager',
    'ArticleVectorManager'
]
