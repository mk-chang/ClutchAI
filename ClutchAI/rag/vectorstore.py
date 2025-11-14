"""
Vectorstore Management for ClutchAI

This module provides utilities to manage various resources (YouTube videos, articles, etc.)
and add them to a ChromaDB vectorstore.

Architecture:
- VectorstoreManager: Orchestrates vectorstore operations and coordinates resource managers
- YoutubeVectorManager: Handles YouTube video ingestion into the vectorstore
- ArticleVectorManager: Handles article ingestion into the vectorstore
- Future managers: Can be added for other resource types (podcasts, etc.)
"""

from __future__ import annotations

import os
import yaml
import requests
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from urllib.parse import urlparse

from scrapy.selector import Selector
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.youtube import TranscriptFormat
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ClutchAI.rag.data_pipelines import (
    BaseVectorManager,
    YoutubeVectorManager,
    ArticleVectorManager,
)
from ClutchAI.rag.data_class import YouTubeVideo


class VectorstoreManager:
    """
    Orchestrator for managing vectorstore operations across multiple resource types.
    
    This class coordinates resource-specific managers (YoutubeVectorManager, ArticleVectorManager, etc.) to
    ingest various types of content into a unified ChromaDB vectorstore.
    
    Features:
    - Manages vectorstore and embeddings
    - Coordinates YouTube video ingestion via YoutubeVectorManager
    - Coordinates article ingestion via ArticleVectorManager
    - Supports YAML-based configuration for multiple resource types
    """
    
    def __init__(
        self,
        vectordata_yaml: Optional[str] = None,
        chroma_persist_directory: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        env_file_location: Optional[Path] = None,
    ):
        """
        Initialize the VectorstoreManager.
        
        Args:
            vectordata_yaml: Path to YAML file storing resource data (default: ClutchAI/vectordata.yaml)
            chroma_persist_directory: Directory for ChromaDB persistence (default: data/chroma_db)
            openai_api_key: OpenAI API key for embeddings (or from env)
            env_file_location: Path to .env file location
        """
        # Set environment file location
        if env_file_location is None:
            self.env_file_location = Path(__file__).parent.parent.resolve()
        else:
            self.env_file_location = Path(env_file_location)
        
        # Set API key
        self.openai_api_key = openai_api_key or os.environ.get('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY env var or pass openai_api_key parameter.")
        
        # Set default paths
        if vectordata_yaml is None:
            self.vectordata_yaml = self.env_file_location / "ClutchAI" / "vectordata.yaml"
        else:
            self.vectordata_yaml = Path(vectordata_yaml)
        
        if chroma_persist_directory is None:
            self.chroma_persist_directory = str(self.env_file_location / "data" / "chroma_db")
        else:
            self.chroma_persist_directory = chroma_persist_directory
        
        # Ensure data directory exists
        Path(self.chroma_persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(api_key=self.openai_api_key)
        
        # Initialize or load vectorstore
        self._vectorstore = None
        self._youtube_manager = None
        self._article_manager = None
    
    def _get_vectorstore(self, create_if_needed: bool = False) -> Optional[Chroma]:
        """
        Get the vectorstore instance if it exists.
        
        Args:
            create_if_needed: Not used, kept for compatibility. Vectorstore is created lazily.
            
        Returns:
            Chroma vectorstore instance if it exists, None otherwise
        """
        if self._vectorstore is None:
            # Check if vectorstore already exists
            if os.path.exists(self.chroma_persist_directory) and os.listdir(self.chroma_persist_directory):
                try:
                    self._vectorstore = Chroma(
                        persist_directory=self.chroma_persist_directory,
                        embedding_function=self.embeddings,
                    )
                except Exception as e:
                    print(f"Warning: Could not load existing vectorstore: {e}")
                    return None
        
        return self._vectorstore
    
    def _get_youtube_manager(self) -> YoutubeVectorManager:
        """
        Get or create the YoutubeVectorManager instance.
        
        Returns:
            YoutubeVectorManager instance
        """
        if self._youtube_manager is None:
            vectorstore = self._get_vectorstore(create_if_needed=False)
            self._youtube_manager = YoutubeVectorManager(
                vectorstore=vectorstore,
                embeddings=self.embeddings,
                persist_directory=self.chroma_persist_directory,
                vectordata_yaml=self.vectordata_yaml
            )
        return self._youtube_manager
    
    def _get_article_manager(self) -> ArticleVectorManager:
        """
        Get or create the ArticleVectorManager instance.
        
        Returns:
            ArticleVectorManager instance
        """
        if self._article_manager is None:
            vectorstore = self._get_vectorstore(create_if_needed=False)
            self._article_manager = ArticleVectorManager(
                vectorstore=vectorstore,
                embeddings=self.embeddings,
                persist_directory=self.chroma_persist_directory,
                vectordata_yaml=self.vectordata_yaml
            )
        return self._article_manager
    
    def add_video_to_vectorstore(
        self,
        url: str,
        chunk_size_seconds: int = 30,
        source_type: str = 'youtube',
        title: Optional[str] = None,
        upload_date: Optional[str] = None,
        publish_date: Optional[str] = None
    ) -> int:
        """
        Add a single video to the vectorstore.
        
        Args:
            url: YouTube video URL
            chunk_size_seconds: Size of transcript chunks in seconds
            source_type: Source type identifier from YAML top-level key (e.g., 'youtube', 'article')
            title: Video title (optional, for metadata)
            upload_date: Upload date in YYYY-MM-DD format (optional, for metadata)
            publish_date: Publish date in YYYY-MM-DD format (optional, for metadata)
            
        Returns:
            Number of document chunks added
        """
        youtube_manager = self._get_youtube_manager()
        return youtube_manager.add_video_to_vectorstore(
            url, 
            chunk_size_seconds,
            source_type=source_type,
            title=title,
            upload_date=upload_date,
            publish_date=publish_date
        )
    
    def add_article_to_vectorstore(
        self,
        url: str,
        source_type: str = 'article',
        title: Optional[str] = None,
        upload_date: Optional[str] = None,
        publish_date: Optional[str] = None
    ) -> int:
        """
        Add a single article to the vectorstore.
        
        Args:
            url: Article URL
            source_type: Source type identifier from YAML top-level key (e.g., 'article')
            title: Article title (optional, will be scraped if not provided)
            upload_date: Upload date in YYYY-MM-DD format (optional, for metadata)
            publish_date: Publish date in YYYY-MM-DD format (optional, for metadata)
            
        Returns:
            Number of document chunks added
        """
        article_manager = self._get_article_manager()
        return article_manager.add_article_to_vectorstore(
            url,
            source_type=source_type,
            title=title,
            upload_date=upload_date,
            publish_date=publish_date
        )
    
    def update_vectorstore(
        self,
        chunk_size_seconds: int = 30,
        skip_existing: bool = True
    ) -> Dict[str, int]:
        """
        Update vectorstore with resources from YAML file.
        
        This method orchestrates updates from all resource managers (YouTube, Article, etc.)
        defined in the YAML configuration file.
        
        Args:
            chunk_size_seconds: Size of transcript chunks in seconds (for YouTube videos)
            skip_existing: Skip resources that are already in the vectorstore (unless force_update is True)
            
        Returns:
            Dictionary with aggregated results from all resource managers
        """
        # Aggregate results from all managers
        aggregated_results = {
            'added': 0,
            'skipped': 0,
            'failed': 0,
            'updated': 0,
            'chunks_added': 0,
            'chunks_deleted': 0
        }
        
        # Process YouTube videos
        youtube_manager = self._get_youtube_manager()
        youtube_results = youtube_manager.update_vectorstore_from_yaml(
            vectordata_yaml=self.vectordata_yaml,
            chunk_size_seconds=chunk_size_seconds,
            skip_existing=skip_existing
        )
        
        # Aggregate YouTube results
        for key in aggregated_results:
            aggregated_results[key] += youtube_results.get(key, 0)
        
        # Process articles
        article_manager = self._get_article_manager()
        article_results = article_manager.update_vectorstore_from_yaml(
            vectordata_yaml=self.vectordata_yaml,
            skip_existing=skip_existing
        )
        
        # Aggregate article results
        for key in aggregated_results:
            aggregated_results[key] += article_results.get(key, 0)
        
        return aggregated_results
    
    def get_vectorstore_stats(self) -> Dict[str, any]:
        """
        Get statistics about the vectorstore.
        
        Returns:
            Dictionary with vectorstore statistics
        """
        stats = {
            'exists': False,
            'document_count': 0,
            'urls_in_vectorstore': 0,
            'youtube_urls': 0,
            'article_urls': 0,
        }
        
        try:
            vectorstore = self._get_vectorstore(create_if_needed=False)
            if vectorstore is not None:
                stats['exists'] = True
                stats['document_count'] = vectorstore._collection.count()
                
                # Get URLs by source type
                try:
                    results = vectorstore._collection.get()
                    if results and 'metadatas' in results:
                        youtube_urls = set()
                        article_urls = set()
                        for metadata in results['metadatas']:
                            if metadata and 'url' in metadata:
                                url = metadata['url']
                                source_type = metadata.get('source_type', '')
                                if source_type == 'youtube':
                                    youtube_urls.add(url)
                                elif source_type == 'article':
                                    article_urls.add(url)
                        
                        stats['youtube_urls'] = len(youtube_urls)
                        stats['article_urls'] = len(article_urls)
                        stats['urls_in_vectorstore'] = len(youtube_urls) + len(article_urls)
                except Exception as e:
                    print(f"Warning: Error getting URL stats: {e}")
                    # Fallback to simple count
                    youtube_manager = self._get_youtube_manager()
                    stats['urls_in_vectorstore'] = len(youtube_manager.get_existing_urls())
        except Exception as e:
            stats['error'] = str(e)
        
        return stats
    
    def get_vectorstore(self) -> Optional[Chroma]:
        """
        Get the vectorstore instance if it exists.
        
        Returns:
            Chroma vectorstore instance if it exists, None otherwise
        """
        return self._get_vectorstore(create_if_needed=False)