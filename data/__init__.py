"""
Cloud SQL PostgreSQL Data Infrastructure

This module provides Cloud SQL PostgreSQL connection management and database utilities.
"""

from data.cloud_sql.connection import PostgresConnection
from data.cloud_sql.schema import setup_schema, create_vector_table
from data.cloud_sql.vector_managers import (
    BaseVectorManager,
    YoutubeVectorManager,
    YoutubeChannelVectorManager,
    ArticleVectorManager
)
from data.data_class import YouTubeVideo, extract_youtube_video_id

__all__ = [
    'PostgresConnection',
    'setup_schema',
    'create_vector_table',
    'BaseVectorManager',
    'YoutubeVectorManager',
    'YoutubeChannelVectorManager',
    'ArticleVectorManager',
    'YouTubeVideo',
    'extract_youtube_video_id'
]
