"""
Cloud SQL PostgreSQL Data Infrastructure

This module provides Cloud SQL PostgreSQL connection management and database utilities.
"""

from data.cloud_sql.connection import PostgresConnection
from data.cloud_sql.schema import setup_pgvector_extension, setup_schema, create_vector_table
from data.cloud_sql.vector_managers import (
    BaseVectorManager,
    YoutubeVectorManager,
    YoutubeChannelVectorManager,
    ArticleVectorManager
)
from data.data_class import YouTubeVideo, extract_youtube_video_id

__all__ = [
    'PostgresConnection',
    'setup_pgvector_extension',
    'setup_schema',  # Deprecated, use setup_pgvector_extension
    'create_vector_table',  # Deprecated, PGVector manages its own tables
    'BaseVectorManager',
    'YoutubeVectorManager',
    'YoutubeChannelVectorManager',
    'ArticleVectorManager',
    'YouTubeVideo',
    'extract_youtube_video_id'
]
