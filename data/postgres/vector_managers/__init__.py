"""
Vector managers package for PostgreSQL with pgvector.

This package provides managers for ingesting different types of content
into a PostgreSQL vectorstore, including YouTube videos and articles.
"""

from data.postgres.vector_managers.base import BaseVectorManager
from data.postgres.vector_managers.youtube import (
    YoutubeVectorManager,
    YoutubeChannelVectorManager
)
from data.postgres.vector_managers.article import ArticleVectorManager

__all__ = [
    'BaseVectorManager',
    'YoutubeVectorManager',
    'YoutubeChannelVectorManager',
    'ArticleVectorManager',
]
