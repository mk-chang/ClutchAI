from __future__ import annotations

import os
import yaml
import requests
import re
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from urllib.parse import urlparse, parse_qs
from abc import ABC, abstractmethod

from scrapy.selector import Selector

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.youtube import TranscriptFormat
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ClutchAI.rag.data_class import YouTubeVideo, extract_youtube_video_id


class BaseVectorManager(ABC):
    """
    Base class for vectorstore managers.
    
    This class provides common functionality for managing resources in a vectorstore,
    including vectorstore initialization, URL management, document deletion, and
    YAML-based updates.
    """
    
    def __init__(
        self,
        vectorstore: Optional[Chroma],
        embeddings: Optional[OpenAIEmbeddings],
        persist_directory: str,
        vectordata_yaml: Optional[Path] = None,
    ):
        """
        Initialize the BaseVectorManager.
        
        Args:
            vectorstore: Chroma vectorstore instance to use (can be None, will be created on first use)
            embeddings: OpenAIEmbeddings instance for document embedding (optional, required for adding documents)
            persist_directory: Directory path for ChromaDB persistence
            vectordata_yaml: Path to YAML file with resource data (optional, can be passed to methods)
        """
        self._vectorstore = vectorstore
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.vectordata_yaml = vectordata_yaml
    
    @property
    def vectorstore(self) -> Optional[Chroma]:
        """Get the vectorstore instance if it exists."""
        if self._vectorstore is None:
            # Check if vectorstore exists
            if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
                if self.embeddings is None:
                    return None
                try:
                    self._vectorstore = Chroma(
                        persist_directory=self.persist_directory,
                        embedding_function=self.embeddings,
                    )
                except Exception as e:
                    print(f"Warning: Could not load vectorstore: {e}")
                    return None
        return self._vectorstore
    
    def normalize_url(self, url: str) -> str:
        """
        Normalize URL for comparison.
        
        Base implementation removes timestamps. Subclasses can override for additional normalization.
        
        Args:
            url: URL to normalize
            
        Returns:
            Normalized URL
        """
        # Remove timestamp parameters: &t=, #t=, &t (without value), etc.
        normalized = url.split('&t=')[0].split('#t=')[0]
        # Handle &t without = (e.g., &t at end of URL)
        if '&t' in normalized and '&t=' not in normalized:
            normalized = normalized.split('&t')[0]
        return normalized
    
    @classmethod
    def _ensure_string_date(cls, date_value) -> Optional[str]:
        """
        Convert date value to string format for ChromaDB metadata.
        
        ChromaDB only accepts str, int, float, bool, SparseVector, or None for metadata.
        This method ensures dates are converted to strings in YYYY-MM-DD format.
        
        Args:
            date_value: Date value (can be str, date, datetime, or None)
            
        Returns:
            String representation of date in YYYY-MM-DD format, or None
        """
        if date_value is None:
            return None
        
        # If already a string, return as-is
        if isinstance(date_value, str):
            return date_value
        
        # If it's a date or datetime object, convert to string
        if hasattr(date_value, 'strftime'):
            return date_value.strftime('%Y-%m-%d')
        
        # Fallback: convert to string
        return str(date_value)
    
    def get_existing_resource_ids(self) -> Set[str]:
        """
        Get resource IDs of resources already in the vectorstore.
        
        Returns:
            Set of resource IDs
        """
        try:
            vs = self.vectorstore
            if vs is None:
                return set()
            
            # Get all documents from the collection
            results = vs._collection.get()
            
            # Extract unique resource IDs from metadata
            ids = set()
            if results and 'metadatas' in results:
                for metadata in results['metadatas']:
                    if metadata and 'resource_id' in metadata and metadata['resource_id']:
                        ids.add(metadata['resource_id'])
            
            return ids
        except Exception as e:
            print(f"Warning: Could not retrieve existing resource IDs from vectorstore: {e}")
            return set()
    
    def delete_documents_by_identifier(self, resource_id: str) -> int:
        """
        Delete all documents from vectorstore that match the given resource ID.
        
        This method finds documents by matching the 'resource_id' field in metadata.
        
        Args:
            resource_id: Unique resource identifier (required)
            
        Returns:
            Number of documents deleted
        """
        if not resource_id:
            return 0
            
        try:
            vs = self.vectorstore
            if vs is None:
                return 0
            
            # Get all documents from the collection
            results = vs._collection.get()
            
            if not results or 'ids' not in results or 'metadatas' not in results:
                return 0
            
            # Find document IDs that match the resource ID
            ids_to_delete = []
            for doc_id, metadata in zip(results['ids'], results['metadatas']):
                if metadata and 'resource_id' in metadata:
                    if metadata['resource_id'] == resource_id:
                        ids_to_delete.append(doc_id)
            
            # Delete documents
            if ids_to_delete:
                vs._collection.delete(ids=ids_to_delete)
                return len(ids_to_delete)
            
            return 0
        except Exception as e:
            print(f"Warning: Error deleting documents for resource_id {resource_id}: {e}")
            return 0
    
    def _add_documents_to_vectorstore(self, docs: List[Document]) -> None:
        """
        Add documents to the vectorstore, creating it if necessary.
        
        Args:
            docs: List of Document objects to add
        """
        if self.embeddings is None:
            raise ValueError("Embeddings are required to add documents to vectorstore")
        
        try:
            vs = self.vectorstore
            if vs is None:
                # Vectorstore doesn't exist, create it with these documents
                self._vectorstore = Chroma.from_documents(
                    documents=docs,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory,
                )
            else:
                # Check if it's empty
                try:
                    results = vs._collection.get()
                    if not results or not results.get('ids'):
                        # Empty vectorstore, recreate with documents
                        self._vectorstore = Chroma.from_documents(
                            documents=docs,
                            embedding=self.embeddings,
                            persist_directory=self.persist_directory,
                        )
                    else:
                        # Add documents to existing vectorstore
                        vs.add_documents(docs)
                except Exception:
                    # If we can't check, just try to add documents
                    vs.add_documents(docs)
        except Exception as e:
            # Fallback: try to create vectorstore from documents
            print(f"Warning: Error with vectorstore: {e}. Creating new vectorstore...")
            self._vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
            )
    
    @abstractmethod
    def load_resources_from_yaml(self, vectordata_yaml: Optional[Path] = None) -> List[Tuple[str, YouTubeVideo]]:
        """
        Load resources from YAML file.
        
        Args:
            vectordata_yaml: Path to YAML file (uses self.vectordata_yaml if not provided)
            
        Returns:
            List of tuples (source_type, YouTubeVideo) where source_type is the YAML top-level key
        """
        pass
    
    @abstractmethod
    def load_resource_content(
        self,
        url: str,
        source_type: str,
        title: Optional[str] = None,
        upload_date: Optional[str] = None,
        publish_date: Optional[str] = None,
        resource_id: Optional[str] = None,
        **kwargs
    ) -> List[Document]:
        """
        Load and process resource content from a URL.
        
        Args:
            url: Resource URL
            source_type: Source type identifier from YAML top-level key
            title: Resource title (optional, for metadata)
            upload_date: Upload date in YYYY-MM-DD format (optional, for metadata)
            publish_date: Publish date in YYYY-MM-DD format (optional, for metadata)
            resource_id: Unique resource identifier (optional, for matching even if URL changes)
            **kwargs: Additional arguments specific to the resource type
            
        Returns:
            List of Document objects with resource chunks and enhanced metadata
        """
        pass
    
    @abstractmethod
    def add_resource_to_vectorstore(
        self,
        url: str,
        source_type: str,
        title: Optional[str] = None,
        upload_date: Optional[str] = None,
        publish_date: Optional[str] = None,
        resource_id: Optional[str] = None,
        **kwargs
    ) -> int:
        """
        Add a single resource to the vectorstore.
        
        Args:
            url: Resource URL
            source_type: Source type identifier from YAML top-level key
            title: Resource title (optional, for metadata)
            upload_date: Upload date in YYYY-MM-DD format (optional, for metadata)
            publish_date: Publish date in YYYY-MM-DD format (optional, for metadata)
            resource_id: Unique resource identifier (optional, for matching even if URL changes)
            **kwargs: Additional arguments specific to the resource type
            
        Returns:
            Number of document chunks added
        """
        pass
    
    def update_vectorstore_from_yaml(
        self,
        vectordata_yaml: Optional[Path] = None,
        skip_existing: bool = True,
        **kwargs
    ) -> Dict[str, int]:
        """
        Update vectorstore with all resources from YAML file.
        
        Args:
            vectordata_yaml: Path to YAML file (uses self.vectordata_yaml if not provided)
            skip_existing: Skip resources that are already in the vectorstore (unless force_update is True)
            **kwargs: Additional arguments specific to the resource type (e.g., chunk_size_seconds for YouTube)
            
        Returns:
            Dictionary with 'added', 'skipped', 'failed', 'updated', 'chunks_added', 'chunks_deleted' counts
        """
        # Load resources from YAML
        resources_with_source = self.load_resources_from_yaml(vectordata_yaml)
        
        # Get existing resource IDs if skipping
        existing_resource_ids = set()
        if skip_existing:
            existing_resource_ids = self.get_existing_resource_ids()
        
        results = {
            'added': 0,
            'skipped': 0,
            'failed': 0,
            'updated': 0,
            'chunks_added': 0,
            'chunks_deleted': 0
        }
        
        for source_type, resource in resources_with_source:
            url = resource.url
            resource_id = getattr(resource, 'id', None)
            title = resource.title
            upload_date = resource.upload_date
            publish_date = getattr(resource, 'publish_date', None) or upload_date
            force_update = resource.force_update
            
            # Require resource_id for identification
            if not resource_id:
                print(f"Warning: Skipping {title} - resource_id is required but not provided")
                results['failed'] += 1
                continue
            
            # Check if resource already exists in vectorstore
            resource_exists = resource_id in existing_resource_ids
            was_updated = False
            
            # Handle force_update: delete existing documents first
            if force_update and resource_exists:
                print(f"Force update requested for {title}. Deleting existing documents...")
                deleted_count = self.delete_documents_by_identifier(resource_id)
                if deleted_count > 0:
                    results['chunks_deleted'] += deleted_count
                    was_updated = True
                    print(f"  ✓ Deleted {deleted_count} existing chunks")
                # Remove from existing_resource_ids so it will be re-added
                existing_resource_ids.discard(resource_id)
                resource_exists = False
            
            # Skip if resource exists and not forcing update
            if skip_existing and resource_exists and not force_update:
                print(f"Skipping {title} (already in vectorstore, force_update=False)")
                results['skipped'] += 1
                continue
            
            try:
                action = "Updating" if was_updated else "Adding"
                print(f"{action} {title} to vectorstore...")
                chunks_added = self.add_resource_to_vectorstore(
                    url,
                    source_type=source_type,
                    title=title,
                    upload_date=upload_date,
                    publish_date=publish_date,
                    resource_id=resource_id,
                    **kwargs
                )
                
                if chunks_added > 0:
                    if was_updated:
                        # This was a force update - count as updated
                        results['updated'] += 1
                    else:
                        results['added'] += 1
                    results['chunks_added'] += chunks_added
                    # Update existing_resource_ids to avoid duplicates
                    existing_resource_ids.add(resource_id)
                    print(f"  ✓ Added {chunks_added} chunks")
                else:
                    results['failed'] += 1
                    print(f"  ✗ Failed to add (no content)")
            except Exception as e:
                results['failed'] += 1
                print(f"  ✗ Failed to add {title}: {e}")
        
        return results


class YoutubeVectorManager(BaseVectorManager):
    """
    Manager for ingesting YouTube videos into a vectorstore.
    
    This class handles all YouTube-specific operations including:
    - Loading videos from YAML configuration
    - Loading video transcripts
    - Adding videos to vectorstore
    - Deleting videos from vectorstore
    - Handling force updates
    """
    
    def __init__(
        self,
        vectorstore: Optional[Chroma],
        embeddings: OpenAIEmbeddings,
        persist_directory: str,
        vectordata_yaml: Optional[Path] = None,
        chunk_size_seconds: int = 30,
    ):
        """
        Initialize the YoutubeVectorManager.
        
        Args:
            vectorstore: Chroma vectorstore instance to use (can be None, will be created on first use)
            embeddings: OpenAIEmbeddings instance for document embedding
            persist_directory: Directory path for ChromaDB persistence
            vectordata_yaml: Path to YAML file with video data (optional, can be passed to methods)
            chunk_size_seconds: Size of transcript chunks in seconds (default: 30)
        """
        super().__init__(vectorstore, embeddings, persist_directory, vectordata_yaml)
        self.chunk_size_seconds = chunk_size_seconds
    
    def load_resources_from_yaml(self, vectordata_yaml: Optional[Path] = None) -> List[Tuple[str, YouTubeVideo]]:
        """
        Load videos/resources from YAML file.
        
        Args:
            vectordata_yaml: Path to YAML file (uses self.vectordata_yaml if not provided)
            
        Returns:
            List of tuples (source_type, YouTubeVideo) where source_type is the YAML top-level key
            (e.g., 'youtube', 'article')
        """
        yaml_path = vectordata_yaml or self.vectordata_yaml
        if not yaml_path or not yaml_path.exists():
            print(f"Warning: YAML file not found at {yaml_path}")
            return []
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                
                if not data:
                    print("Warning: YAML file is empty.")
                    return []
                
                videos = []
                # Iterate through all top-level keys, but only process 'youtube' key
                for source_type, items in data.items():
                    if source_type != 'youtube' or not isinstance(items, list):
                        continue
                    
                    for item_data in items:
                        if not isinstance(item_data, dict):
                            print(f"Warning: Skipping invalid entry in {source_type}: {item_data}")
                            continue
                        
                        # Ensure required fields
                        if 'title' not in item_data or 'url' not in item_data:
                            print(f"Warning: Skipping {source_type} entry without title or url: {item_data}")
                            continue
                        
                        video = YouTubeVideo.from_dict(item_data)
                        videos.append((source_type, video))
                
                return videos
        except (yaml.YAMLError, KeyError, TypeError) as e:
            print(f"Warning: Error loading YAML file: {e}. Starting with empty list.")
            return []
    
    def load_resource_content(
        self, 
        url: str,
        source_type: str = 'youtube',
        title: Optional[str] = None,
        upload_date: Optional[str] = None,
        publish_date: Optional[str] = None,
        resource_id: Optional[str] = None,
        chunk_size_seconds: Optional[int] = None,
        **kwargs
    ) -> List[Document]:
        """
        Load transcript from a YouTube video URL and enhance with custom metadata.
        
        Args:
            url: YouTube video URL
            source_type: Source type identifier from YAML top-level key (e.g., 'youtube', 'article')
            title: Video title (optional, for metadata)
            upload_date: Upload date in YYYY-MM-DD format (optional, for metadata)
            publish_date: Publish date in YYYY-MM-DD format (optional, for metadata)
            resource_id: Unique resource identifier (optional, auto-extracted from URL if not provided)
            chunk_size_seconds: Size of transcript chunks in seconds
            **kwargs: Additional arguments (unused for YouTube)
            
        Returns:
            List of Document objects with transcript chunks and enhanced metadata
        """
        try:
            # Normalize URL before loading (remove timestamp parameters that can cause errors)
            url_normalized = self.normalize_url(url)
            
            # Use instance default if not provided
            chunk_size = chunk_size_seconds if chunk_size_seconds is not None else self.chunk_size_seconds
            
            loader = YoutubeLoader.from_youtube_url(
                url_normalized,
                add_video_info=False,
                transcript_format=TranscriptFormat.CHUNKS,
                chunk_size_seconds=chunk_size
            )
            docs = loader.load()
            
            # Auto-extract resource_id if not provided
            if not resource_id:
                resource_id = extract_youtube_video_id(url)
            
            # Enhance metadata for all documents
            # NOTE: Adding metadata is computationally lightweight. Metadata is stored as key-value
            # pairs and does not affect embedding computation or search performance. The storage
            # overhead is negligible compared to document content and embeddings. ChromaDB efficiently
            # indexes metadata for fast filtering and retrieval.
            for doc in docs:
                # Ensure metadata dict exists
                if not doc.metadata:
                    doc.metadata = {}
                
                # Add/update custom metadata fields
                doc.metadata['source_type'] = source_type  # Source type from YAML top-level key
                doc.metadata['url'] = url_normalized  # Normalized URL without timestamp
                
                # Add resource_id if available (for matching even if URL changes)
                if resource_id:
                    doc.metadata['resource_id'] = resource_id
                
                # Add title if provided
                if title:
                    doc.metadata['title'] = title
                
                # Add upload_date if provided (ensure it's a string for ChromaDB)
                if upload_date:
                    doc.metadata['upload_date'] = self._ensure_string_date(upload_date)
                
                # Add publish_date if provided (defaults to upload_date if not provided)
                if publish_date:
                    doc.metadata['publish_date'] = self._ensure_string_date(publish_date)
                elif upload_date:
                    doc.metadata['publish_date'] = self._ensure_string_date(upload_date)
                
                # Keep existing metadata from YoutubeLoader:
                # - 'source': Full URL with timestamp (e.g., https://youtube.com/watch?v=ID&t=30s)
                #   Useful for linking back to specific moments in the video
                # - 'start_seconds': Start time of the chunk in seconds
                # - 'start_timestamp': Human-readable timestamp (e.g., '00:00:30')
            
            return docs
        except Exception as e:
            raise ValueError(f"Failed to load transcript from {url}: {e}") from e
    
    def add_resource_to_vectorstore(
        self,
        url: str,
        source_type: str = 'youtube',
        title: Optional[str] = None,
        upload_date: Optional[str] = None,
        publish_date: Optional[str] = None,
        resource_id: Optional[str] = None,
        chunk_size_seconds: Optional[int] = None,
        **kwargs
    ) -> int:
        """
        Add a single video to the vectorstore.
        
        Args:
            url: YouTube video URL
            source_type: Source type identifier from YAML top-level key (e.g., 'youtube', 'article')
            title: Video title (optional, for metadata)
            upload_date: Upload date in YYYY-MM-DD format (optional, for metadata)
            publish_date: Publish date in YYYY-MM-DD format (optional, for metadata)
            resource_id: Unique resource identifier (optional, auto-extracted from URL if not provided)
            chunk_size_seconds: Size of transcript chunks in seconds
            **kwargs: Additional arguments (unused for YouTube)
            
        Returns:
            Number of document chunks added
        """
        # Load transcript with enhanced metadata
        docs = self.load_resource_content(
            url,
            source_type=source_type,
            title=title,
            upload_date=upload_date,
            publish_date=publish_date,
            resource_id=resource_id,
            chunk_size_seconds=chunk_size_seconds,
            **kwargs
        )
        
        if not docs:
            print(f"Warning: No transcript found for {url}")
            return 0
        
        # Add documents to vectorstore using base class method
        self._add_documents_to_vectorstore(docs)
        
        return len(docs)
    
    # Keep the old method name for backward compatibility
    def load_video_transcript(
        self, 
        url: str, 
        chunk_size_seconds: Optional[int] = None,
        source_type: str = 'youtube',
        title: Optional[str] = None,
        upload_date: Optional[str] = None,
        publish_date: Optional[str] = None
    ) -> List[Document]:
        """Backward compatibility wrapper for load_resource_content."""
        return self.load_resource_content(
            url,
            source_type=source_type,
            title=title,
            upload_date=upload_date,
            publish_date=publish_date,
            chunk_size_seconds=chunk_size_seconds
        )
    
    def add_video_to_vectorstore(
        self,
        url: str,
        chunk_size_seconds: Optional[int] = None,
        source_type: str = 'youtube',
        title: Optional[str] = None,
        upload_date: Optional[str] = None,
        publish_date: Optional[str] = None
    ) -> int:
        """Backward compatibility wrapper for add_resource_to_vectorstore."""
        return self.add_resource_to_vectorstore(
            url,
            source_type=source_type,
            title=title,
            upload_date=upload_date,
            publish_date=publish_date,
            chunk_size_seconds=chunk_size_seconds
        )
    
    def load_videos_from_yaml(self, vectordata_yaml: Optional[Path] = None) -> List[Tuple[str, YouTubeVideo]]:
        """Backward compatibility wrapper for load_resources_from_yaml."""
        return self.load_resources_from_yaml(vectordata_yaml)
    
    def update_vectorstore_from_yaml(
        self,
        vectordata_yaml: Optional[Path] = None,
        chunk_size_seconds: Optional[int] = None,
        skip_existing: bool = True
    ) -> Dict[str, int]:
        """
        Update vectorstore with all videos from YAML file.
        
        Args:
            vectordata_yaml: Path to YAML file (uses self.vectordata_yaml if not provided)
            chunk_size_seconds: Size of transcript chunks in seconds (uses instance default if not provided)
            skip_existing: Skip videos that are already in the vectorstore (unless force_update is True)
            
        Returns:
            Dictionary with 'added', 'skipped', 'failed', 'updated', 'chunks_added', 'chunks_deleted' counts
        """
        # Use instance default if not provided
        chunk_size = chunk_size_seconds if chunk_size_seconds is not None else self.chunk_size_seconds
        
        # Call base class method with chunk_size_seconds as kwargs
        return super().update_vectorstore_from_yaml(
            vectordata_yaml=vectordata_yaml,
            skip_existing=skip_existing,
            chunk_size_seconds=chunk_size
        )


class YoutubeChannelVectorManager(YoutubeVectorManager):
    """
    Manager for ingesting entire YouTube channels into a vectorstore.
    
    This class extends YoutubeVectorManager with channel-level operations including:
    - Fetching all videos from a YouTube channel
    - Processing channel videos in bulk
    - Estimating processing requirements
    - Adding channel videos to vectorstore
    """
    
    def fetch_channel_videos(
        self,
        channel_handle: str,
        youtube_api_key: str,
        published_after: Optional[str] = None,
        published_before: Optional[str] = None,
        max_results: Optional[int] = None
    ) -> List[Dict[str, any]]:
        """
        Fetch all videos from a YouTube channel using YouTube Data API v3.
        
        Args:
            channel_handle: YouTube channel handle (e.g., '@LockedOnFantasyBasketball' or 'UC...')
            youtube_api_key: YouTube Data API v3 key
            published_after: Filter videos published after this date (ISO 8601 format: YYYY-MM-DD)
            published_before: Filter videos published before this date (ISO 8601 format: YYYY-MM-DD)
            max_results: Maximum number of videos to fetch (None for all)
            
        Returns:
            List of dictionaries with video information (id, title, url, publishedAt, etc.)
        """
        try:
            from googleapiclient.discovery import build
            from googleapiclient.errors import HttpError
        except ImportError:
            raise ImportError(
                "google-api-python-client is required. Install it with: pip install google-api-python-client"
            )
        
        youtube = build('youtube', 'v3', developerKey=youtube_api_key)
        videos = []
        next_page_token = None
        
        # Convert channel handle to channel ID if needed
        channel_id = None
        if channel_handle.startswith('@'):
            # Handle format: @channelname - extract the handle name
            handle_name = channel_handle[1:]  # Remove @
            
            # Try to get channel by custom URL (handle)
            try:
                # First try using search API
                search_response = youtube.search().list(
                    q=channel_handle,
                    type='channel',
                    part='id',
                    maxResults=1
                ).execute()
                
                if search_response.get('items'):
                    channel_id = search_response['items'][0]['id']['channelId']
                else:
                    # Fallback: try channels().list with forUsername (legacy, may not work for @ handles)
                    # For @ handles, we need to use the custom URL format
                    # The handle maps to youtube.com/@handle, but API doesn't directly support this
                    # So we'll use search as primary method
                    raise ValueError(f"Channel not found: {channel_handle}. Make sure the handle is correct.")
            except HttpError as e:
                raise ValueError(f"Error finding channel {channel_handle}: {e}")
            except ValueError as e:
                raise
        elif channel_handle.startswith('UC') and len(channel_handle) == 24:
            # Already a channel ID (starts with UC and is 24 chars)
            channel_id = channel_handle
        else:
            # Try to use as channel ID or custom URL
            # If it contains /, it might be a full URL
            if '/' in channel_handle:
                # Extract from URL like https://www.youtube.com/@channelname
                import re
                match = re.search(r'@([^/?]+)', channel_handle)
                if match:
                    handle_name = match.group(1)
                    # Use search to find channel
                    search_response = youtube.search().list(
                        q=f"@{handle_name}",
                        type='channel',
                        part='id',
                        maxResults=1
                    ).execute()
                    if search_response.get('items'):
                        channel_id = search_response['items'][0]['id']['channelId']
                    else:
                        raise ValueError(f"Channel not found in URL: {channel_handle}")
                else:
                    raise ValueError(f"Could not extract channel handle from URL: {channel_handle}")
            else:
                # Assume it's a channel ID
                channel_id = channel_handle
        
        # Build search parameters
        search_params = {
            'channelId': channel_id,
            'type': 'video',
            'part': 'id',
            'order': 'date',
            'maxResults': 50  # Maximum per request
        }
        
        if published_after:
            search_params['publishedAfter'] = f"{published_after}T00:00:00Z"
        if published_before:
            search_params['publishedBefore'] = f"{published_before}T23:59:59Z"
        
        # Fetch all videos
        while True:
            if next_page_token:
                search_params['pageToken'] = next_page_token
            
            try:
                search_response = youtube.search().list(**search_params).execute()
            except HttpError as e:
                raise ValueError(f"YouTube API error: {e}")
            
            video_ids = [item['id']['videoId'] for item in search_response.get('items', [])]
            
            if not video_ids:
                break
            
            # Get detailed video information
            video_details = youtube.videos().list(
                part='snippet,contentDetails,statistics',
                id=','.join(video_ids)
            ).execute()
            
            for video in video_details.get('items', []):
                video_info = {
                    'id': video['id'],
                    'title': video['snippet']['title'],
                    'description': video['snippet'].get('description', ''),
                    'publishedAt': video['snippet']['publishedAt'],
                    'url': f"https://www.youtube.com/watch?v={video['id']}",
                    'duration': video['contentDetails'].get('duration', ''),
                    'viewCount': int(video['statistics'].get('viewCount', 0)),
                }
                videos.append(video_info)
                
                if max_results and len(videos) >= max_results:
                    return videos
            
            # Check for next page
            next_page_token = search_response.get('nextPageToken')
            if not next_page_token:
                break
        
        return videos
    
    def _estimate_processing_requirements(
        self,
        videos: List[Dict[str, any]],
        chunk_size_seconds: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Estimate time and memory requirements for processing YouTube videos.
        
        Args:
            videos: List of video dictionaries with duration and other info
            chunk_size_seconds: Size of transcript chunks in seconds (uses instance default if not provided)
            
        Returns:
            Dictionary with estimated_time_minutes and estimated_memory_mb
        """
        import re
        from datetime import timedelta
        
        chunk_size = chunk_size_seconds if chunk_size_seconds is not None else self.chunk_size_seconds
        
        def parse_duration(duration_str: str) -> int:
            """Parse ISO 8601 duration (e.g., PT1H30M15S) to seconds."""
            if not duration_str:
                return 0
            pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
            match = re.match(pattern, duration_str)
            if not match:
                return 0
            hours = int(match.group(1) or 0)
            minutes = int(match.group(2) or 0)
            seconds = int(match.group(3) or 0)
            return hours * 3600 + minutes * 60 + seconds
        
        # Estimate based on video durations
        total_duration_seconds = 0
        total_videos = len(videos)
        
        for video in videos:
            duration_seconds = parse_duration(video.get('duration', ''))
            if duration_seconds == 0:
                # Default estimate: 30 minutes per video if duration unknown
                duration_seconds = 30 * 60
            total_duration_seconds += duration_seconds
        
        # Estimate chunks
        # Average podcast is ~30-60 minutes, chunked every 30 seconds
        # So roughly 60-120 chunks per video
        estimated_chunks_per_video = total_duration_seconds / chunk_size if total_videos > 0 else 0
        total_chunks = estimated_chunks_per_video * total_videos
        
        # Time estimates (in seconds per operation)
        # - YouTube API fetch: ~0.5s per video (batch requests)
        # - Transcript download: ~2-5s per video
        # - Text chunking: ~0.1s per chunk
        # - Embedding generation: ~0.5s per chunk (OpenAI API)
        # - Vectorstore insertion: ~0.1s per chunk
        
        api_fetch_time = total_videos * 0.5
        transcript_time = total_videos * 3.5  # Average
        chunking_time = total_chunks * 0.1
        embedding_time = total_chunks * 0.5  # OpenAI API rate limit considered
        insertion_time = total_chunks * 0.1
        
        # Add 20% overhead for network delays, retries, etc.
        total_time_seconds = (api_fetch_time + transcript_time + chunking_time + 
                             embedding_time + insertion_time) * 1.2
        
        estimated_time_minutes = total_time_seconds / 60
        
        # Memory estimates (in MB)
        # - Each chunk: ~1-2 KB text + ~1.5 KB embedding (1536 dims * 4 bytes) = ~7 KB per chunk
        # - ChromaDB overhead: ~20% of data size
        # - Python objects overhead: ~30% of data size
        
        bytes_per_chunk = 7 * 1024  # 7 KB
        total_data_mb = (total_chunks * bytes_per_chunk) / (1024 * 1024)
        estimated_memory_mb = total_data_mb * 1.5  # 50% overhead for ChromaDB and Python
        
        return {
            'estimated_time_minutes': estimated_time_minutes,
            'estimated_memory_mb': estimated_memory_mb,
            'estimated_chunks': int(total_chunks),
            'estimated_total_duration_hours': total_duration_seconds / 3600
        }
    
    def add_channel_to_vectorstore(
        self,
        channel_handle: str,
        youtube_api_key: str,
        season_start: str = "2025-07-01",
        season_end: str = "2026-06-30",
        chunk_size_seconds: Optional[int] = None,
        skip_existing: bool = True,
        estimate_only: bool = False
    ) -> Dict[str, any]:
        """
        Pipeline to fetch all videos from a YouTube channel published during NBA season and add to vectorstore.
        
        Args:
            channel_handle: YouTube channel handle (e.g., '@LockedOnFantasyBasketball')
            youtube_api_key: YouTube Data API v3 key
            season_start: Start date of NBA season (YYYY-MM-DD format, default: 2025-07-01)
            season_end: End date of NBA season (YYYY-MM-DD format, default: 2026-06-30)
            chunk_size_seconds: Size of transcript chunks in seconds (uses instance default if not provided)
            skip_existing: Skip videos already in vectorstore
            estimate_only: If True, only return estimates without processing videos
            
        Returns:
            Dictionary with results including:
            - videos_found: Number of videos found
            - videos_added: Number of videos added
            - videos_skipped: Number of videos skipped
            - videos_failed: Number of videos that failed
            - chunks_added: Total chunks added
            - estimated_time_minutes: Estimated processing time
            - estimated_memory_mb: Estimated memory usage
        """
        from datetime import datetime, timedelta
        
        chunk_size = chunk_size_seconds if chunk_size_seconds is not None else self.chunk_size_seconds
        
        print(f"Fetching videos from channel: {channel_handle}")
        print(f"Season range: {season_start} to {season_end}")
        
        # Fetch videos from channel
        videos = self.fetch_channel_videos(
            channel_handle=channel_handle,
            youtube_api_key=youtube_api_key,
            published_after=season_start,
            published_before=season_end
        )
        
        print(f"Found {len(videos)} videos in season range")
        
        if estimate_only:
            return self._estimate_processing_requirements(videos, chunk_size)
        
        # Get existing resource IDs if skipping
        existing_resource_ids = set()
        if skip_existing:
            existing_resource_ids = self.get_existing_resource_ids()
        
        results = {
            'videos_found': len(videos),
            'videos_added': 0,
            'videos_skipped': 0,
            'videos_failed': 0,
            'chunks_added': 0,
            'estimated_time_minutes': 0,
            'estimated_memory_mb': 0
        }
        
        # Calculate estimates
        estimates = self._estimate_processing_requirements(videos, chunk_size)
        results['estimated_time_minutes'] = estimates['estimated_time_minutes']
        results['estimated_memory_mb'] = estimates['estimated_memory_mb']
        
        print(f"\nEstimated processing time: {results['estimated_time_minutes']:.1f} minutes")
        print(f"Estimated memory usage: {results['estimated_memory_mb']:.1f} MB")
        print(f"\nProcessing videos...")
        
        # Process each video
        for i, video in enumerate(videos, 1):
            video_id = video['id']
            video_url = video['url']
            video_title = video['title']
            published_at = video['publishedAt']
            
            # Parse publish date
            publish_date = datetime.fromisoformat(published_at.replace('Z', '+00:00')).strftime('%Y-%m-%d')
            
            print(f"\n[{i}/{len(videos)}] Processing: {video_title}")
            
            # Check if already exists
            if skip_existing and video_id in existing_resource_ids:
                print(f"  ⏭ Skipping (already in vectorstore)")
                results['videos_skipped'] += 1
                continue
            
            try:
                chunks_added = self.add_resource_to_vectorstore(
                    url=video_url,
                    source_type='youtube',
                    title=video_title,
                    upload_date=publish_date,
                    publish_date=publish_date,
                    resource_id=video_id,
                    chunk_size_seconds=chunk_size
                )
                
                if chunks_added > 0:
                    results['videos_added'] += 1
                    results['chunks_added'] += chunks_added
                    existing_resource_ids.add(video_id)
                    print(f"  ✓ Added {chunks_added} chunks")
                else:
                    results['videos_failed'] += 1
                    print(f"  ✗ Failed (no transcript found)")
            except Exception as e:
                results['videos_failed'] += 1
                print(f"  ✗ Failed: {e}")
        
        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"  Videos found: {results['videos_found']}")
        print(f"  Videos added: {results['videos_added']}")
        print(f"  Videos skipped: {results['videos_skipped']}")
        print(f"  Videos failed: {results['videos_failed']}")
        print(f"  Total chunks added: {results['chunks_added']}")
        print(f"  Estimated time: {results['estimated_time_minutes']:.1f} minutes")
        print(f"  Estimated memory: {results['estimated_memory_mb']:.1f} MB")
        print(f"{'='*60}")
        
        return results


class ArticleVectorManager(BaseVectorManager):
    """
    Manager for ingesting articles into a vectorstore.
    
    This class handles all article-specific operations including:
    - Loading articles from YAML configuration
    - Scraping article content from URLs
    - Chunking article content
    - Adding articles to vectorstore
    - Deleting articles from vectorstore
    - Handling force updates
    """
    
    def __init__(
        self,
        vectorstore: Optional[Chroma],
        embeddings: OpenAIEmbeddings,
        persist_directory: str,
        vectordata_yaml: Optional[Path] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize the ArticleVectorManager.
        
        Args:
            vectorstore: Chroma vectorstore instance to use (can be None, will be created on first use)
            embeddings: OpenAIEmbeddings instance for document embedding
            persist_directory: Directory path for ChromaDB persistence
            vectordata_yaml: Path to YAML file with article data (optional, can be passed to methods)
            chunk_size: Size of text chunks in characters (default: 1000)
            chunk_overlap: Overlap between chunks in characters (default: 200)
        """
        super().__init__(vectorstore, embeddings, persist_directory, vectordata_yaml)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def normalize_url(self, url: str) -> str:
        """
        Normalize URL for comparison.
        
        Overrides base implementation to also strip trailing slashes for articles.
        
        Args:
            url: URL to normalize
            
        Returns:
            Normalized URL
        """
        normalized = super().normalize_url(url)
        return normalized.rstrip('/')
    
    def load_resources_from_yaml(self, vectordata_yaml: Optional[Path] = None) -> List[Tuple[str, YouTubeVideo]]:
        """
        Load articles from YAML file.
        
        Args:
            vectordata_yaml: Path to YAML file (uses self.vectordata_yaml if not provided)
            
        Returns:
            List of tuples (source_type, YouTubeVideo) where source_type is the YAML top-level key
            (e.g., 'article'). Note: Uses YouTubeVideo dataclass as it has the same structure.
        """
        yaml_path = vectordata_yaml or self.vectordata_yaml
        if not yaml_path or not yaml_path.exists():
            print(f"Warning: YAML file not found at {yaml_path}")
            return []
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                
                if not data:
                    print("Warning: YAML file is empty.")
                    return []
                
                articles = []
                # Iterate through all top-level keys, but only process 'article' key
                for source_type, items in data.items():
                    if source_type != 'article' or not isinstance(items, list):
                        continue
                    
                    for item_data in items:
                        if not isinstance(item_data, dict):
                            print(f"Warning: Skipping invalid entry in {source_type}: {item_data}")
                            continue
                        
                        # Ensure required fields
                        if 'title' not in item_data or 'url' not in item_data:
                            print(f"Warning: Skipping {source_type} entry without title or url: {item_data}")
                            continue
                        
                        article = YouTubeVideo.from_dict(item_data)  # Reuse same dataclass
                        articles.append((source_type, article))
                
                return articles
        except (yaml.YAMLError, KeyError, TypeError) as e:
            print(f"Warning: Error loading YAML file: {e}. Starting with empty list.")
            return []
    
    def scrape_article_content(self, url: str, max_content_length: Optional[int] = None) -> Dict[str, str]:
        """
        Scrape article content from a URL.
        
        Args:
            url: Article URL to scrape
            max_content_length: Maximum length of content to return (optional)
            
        Returns:
            Dictionary with 'url', 'title', 'content', and 'source' keys, or None if error
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            # Parse and normalize URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme:
                # If no scheme, assume https
                url = f"https://{url}"
                parsed_url = urlparse(url)
            elif not parsed_url.netloc:
                raise ValueError(f"Invalid URL format: {url}")
            
            # Get the domain for source tracking
            domain = parsed_url.netloc
            
            # Fetch the page
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse with scrapy selector
            selector = Selector(text=response.text)
            
            # Extract title - try multiple common patterns
            title = None
            title_selectors = [
                'h1::text',
                '.article-title::text',
                '.post-title::text',
                '.entry-title::text',
                'title::text',
                'meta[property="og:title"]::attr(content)',
                'meta[name="twitter:title"]::attr(content)'
            ]
            
            for selector_pattern in title_selectors:
                title = selector.css(selector_pattern).get()
                if title:
                    title = title.strip()
                    break
            
            # Fallback to page title tag
            if not title:
                title = selector.css('title::text').get()
                if title:
                    # Clean up title (remove site name if present)
                    title = title.split('|')[0].split('-')[0].strip()
            
            title = title or 'Untitled'
            
            # Extract article content - try multiple common content selectors
            content = None
            content_selectors = [
                'article p::text',
                'article::text',
                '.article-body p::text',
                '.article-content p::text',
                '.post-content p::text',
                '.entry-content p::text',
                '.content p::text',
                'main article p::text',
                'main p::text',
                '[role="article"] p::text'
            ]
            
            for selector_pattern in content_selectors:
                paragraphs = selector.css(selector_pattern).getall()
                if paragraphs:
                    text_content = ' '.join([p.strip() for p in paragraphs if p.strip()])
                    if len(text_content) > 200:  # Only use if substantial content
                        content = text_content
                        break
            
            # Fallback: try to get text from main content areas
            if not content:
                main_content = selector.css('main, article, .main-content, .content').get()
                if main_content:
                    main_selector = Selector(text=main_content)
                    content = ' '.join(main_selector.css('::text').getall())
            
            # Last resort: get all paragraph text from body
            if not content:
                # Use XPath to exclude common non-content elements
                paragraphs = selector.xpath('//body//p[not(ancestor::script|ancestor::style|ancestor::nav|ancestor::header|ancestor::footer|ancestor::aside)]//text()').getall()
                content = ' '.join([p.strip() for p in paragraphs if p.strip()])
            
            # Clean up content
            if content:
                content = ' '.join(content.split())  # Normalize whitespace
                content = content.strip()
            
            if not content or len(content) < 50:
                raise ValueError(f"Minimal or no content extracted from {url}")
            
            # Limit content length if specified
            if max_content_length and len(content) > max_content_length:
                content = content[:max_content_length] + "..."
            
            return {
                'url': url,
                'title': title,
                'content': content,
                'source': domain
            }
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error fetching {url}: {e}") from e
        except Exception as e:
            raise ValueError(f"Error scraping {url}: {e}") from e
    
    def load_resource_content(
        self,
        url: str,
        source_type: str = 'article',
        title: Optional[str] = None,
        upload_date: Optional[str] = None,
        publish_date: Optional[str] = None,
        resource_id: Optional[str] = None,
        **kwargs
    ) -> List[Document]:
        """
        Load and chunk article content from a URL.
        
        Args:
            url: Article URL
            source_type: Source type identifier from YAML top-level key (e.g., 'article')
            title: Article title (optional, will be scraped if not provided)
            upload_date: Upload date in YYYY-MM-DD format (optional, for metadata)
            publish_date: Publish date in YYYY-MM-DD format (optional, for metadata)
            resource_id: Unique resource identifier (optional, should be provided in YAML for articles)
            **kwargs: Additional arguments (unused for articles)
            
        Returns:
            List of Document objects with article chunks and enhanced metadata
        """
        # Scrape article content
        article_data = self.scrape_article_content(url)
        
        # Use scraped title if not provided
        if not title:
            title = article_data['title']
        
        # Normalize URL
        url_normalized = self.normalize_url(url)
        
        # Create initial document
        initial_doc = Document(
            page_content=article_data['content'],
            metadata={
                'source': url_normalized,
                'title': title,
                'source_domain': article_data['source']
            }
        )
        
        # Split document into chunks
        docs = self.text_splitter.split_documents([initial_doc])
        
        # Enhance metadata for all chunks
        for doc in docs:
            # Add/update custom metadata fields
            doc.metadata['source_type'] = source_type
            doc.metadata['url'] = url_normalized
            
            # Add resource_id if available (for matching even if URL changes)
            if resource_id:
                doc.metadata['resource_id'] = resource_id
            
            # Add title
            if title:
                doc.metadata['title'] = title
            
            # Add upload_date if provided (ensure it's a string for ChromaDB)
            if upload_date:
                doc.metadata['upload_date'] = self._ensure_string_date(upload_date)
            
            # Add publish_date if provided (defaults to upload_date if not provided)
            if publish_date:
                doc.metadata['publish_date'] = self._ensure_string_date(publish_date)
            elif upload_date:
                doc.metadata['publish_date'] = self._ensure_string_date(upload_date)
            
            # Keep source_domain from scraping
            if 'source_domain' in doc.metadata:
                # Rename to be consistent, or keep both
                pass
        
        return docs
    
    def add_resource_to_vectorstore(
        self,
        url: str,
        source_type: str = 'article',
        title: Optional[str] = None,
        upload_date: Optional[str] = None,
        publish_date: Optional[str] = None,
        resource_id: Optional[str] = None,
        **kwargs
    ) -> int:
        """
        Add a single article to the vectorstore.
        
        Args:
            url: Article URL
            source_type: Source type identifier from YAML top-level key (e.g., 'article')
            title: Article title (optional, will be scraped if not provided)
            upload_date: Upload date in YYYY-MM-DD format (optional, for metadata)
            publish_date: Publish date in YYYY-MM-DD format (optional, for metadata)
            resource_id: Unique resource identifier (optional, should be provided in YAML for articles)
            **kwargs: Additional arguments (unused for articles)
            
        Returns:
            Number of document chunks added
        """
        # Load article content with enhanced metadata
        docs = self.load_resource_content(
            url,
            source_type=source_type,
            title=title,
            upload_date=upload_date,
            publish_date=publish_date,
            resource_id=resource_id,
            **kwargs
        )
        
        if not docs:
            print(f"Warning: No content found for {url}")
            return 0
        
        # Add documents to vectorstore using base class method
        self._add_documents_to_vectorstore(docs)
        
        return len(docs)
    
    # Keep the old method names for backward compatibility
    def load_articles_from_yaml(self, vectordata_yaml: Optional[Path] = None) -> List[Tuple[str, YouTubeVideo]]:
        """Backward compatibility wrapper for load_resources_from_yaml."""
        return self.load_resources_from_yaml(vectordata_yaml)
    
    def load_article_content(
        self,
        url: str,
        source_type: str = 'article',
        title: Optional[str] = None,
        upload_date: Optional[str] = None,
        publish_date: Optional[str] = None
    ) -> List[Document]:
        """Backward compatibility wrapper for load_resource_content."""
        return self.load_resource_content(
            url,
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
        """Backward compatibility wrapper for add_resource_to_vectorstore."""
        return self.add_resource_to_vectorstore(
            url,
            source_type=source_type,
            title=title,
            upload_date=upload_date,
            publish_date=publish_date
        )
    
    def update_vectorstore_from_yaml(
        self,
        vectordata_yaml: Optional[Path] = None,
        skip_existing: bool = True
    ) -> Dict[str, int]:
        """
        Update vectorstore with all articles from YAML file.
        
        Args:
            vectordata_yaml: Path to YAML file (uses self.vectordata_yaml if not provided)
            skip_existing: Skip articles that are already in the vectorstore (unless force_update is True)
            
        Returns:
            Dictionary with 'added', 'skipped', 'failed', 'updated', 'chunks_added', 'chunks_deleted' counts
        """
        # Call base class method
        return super().update_vectorstore_from_yaml(
            vectordata_yaml=vectordata_yaml,
            skip_existing=skip_existing
        )


# Backward compatibility functions - these create a YoutubeChannelVectorManager instance
# and delegate to the class methods. Prefer using YoutubeChannelVectorManager directly.

def fetch_youtube_channel_videos(
    channel_handle: str,
    youtube_api_key: str,
    published_after: Optional[str] = None,
    published_before: Optional[str] = None,
    max_results: Optional[int] = None
) -> List[Dict[str, any]]:
    """
    Fetch all videos from a YouTube channel using YouTube Data API v3.
    
    This is a backward compatibility wrapper. For new code, prefer using
    YoutubeChannelVectorManager.fetch_channel_videos() directly.
    
    Args:
        channel_handle: YouTube channel handle (e.g., '@LockedOnFantasyBasketball' or 'UC...')
        youtube_api_key: YouTube Data API v3 key
        published_after: Filter videos published after this date (ISO 8601 format: YYYY-MM-DD)
        published_before: Filter videos published before this date (ISO 8601 format: YYYY-MM-DD)
        max_results: Maximum number of videos to fetch (None for all)
        
    Returns:
        List of dictionaries with video information (id, title, url, publishedAt, etc.)
    """
    # Create a minimal manager instance just for fetching videos
    # We don't need vectorstore or embeddings for this operation
    manager = YoutubeChannelVectorManager(
        vectorstore=None,
        embeddings=None,  # Not needed for fetching videos
        persist_directory="",  # Not needed for fetching videos
        chunk_size_seconds=30
    )
    return manager.fetch_channel_videos(
        channel_handle=channel_handle,
        youtube_api_key=youtube_api_key,
        published_after=published_after,
        published_before=published_before,
        max_results=max_results
    )


def add_youtube_channel_to_vectorstore(
    channel_handle: str,
    youtube_api_key: str,
    youtube_manager: YoutubeVectorManager,
    season_start: str = "2025-07-01",
    season_end: str = "2026-06-30",
    chunk_size_seconds: int = 30,
    skip_existing: bool = True,
    estimate_only: bool = False
) -> Dict[str, any]:
    """
    Pipeline to fetch all videos from a YouTube channel published during NBA season and add to vectorstore.
    
    This is a backward compatibility wrapper. For new code, prefer using
    YoutubeChannelVectorManager.add_channel_to_vectorstore() directly.
    
    Args:
        channel_handle: YouTube channel handle (e.g., '@LockedOnFantasyBasketball')
        youtube_api_key: YouTube Data API v3 key
        youtube_manager: YoutubeVectorManager instance (will be converted to YoutubeChannelVectorManager)
        season_start: Start date of NBA season (YYYY-MM-DD format, default: 2025-07-01)
        season_end: End date of NBA season (YYYY-MM-DD format, default: 2026-06-30)
        chunk_size_seconds: Size of transcript chunks in seconds
        skip_existing: Skip videos already in vectorstore
        estimate_only: If True, only return estimates without processing videos
        
    Returns:
        Dictionary with results including:
        - videos_found: Number of videos found
        - videos_added: Number of videos added
        - videos_skipped: Number of videos skipped
        - videos_failed: Number of videos that failed
        - chunks_added: Total chunks added
        - estimated_time_minutes: Estimated processing time
        - estimated_memory_mb: Estimated memory usage
    """
    # Convert YoutubeVectorManager to YoutubeChannelVectorManager
    # by creating a new instance with the same configuration
    channel_manager = YoutubeChannelVectorManager(
        vectorstore=youtube_manager.vectorstore,
        embeddings=youtube_manager.embeddings,
        persist_directory=youtube_manager.persist_directory,
        vectordata_yaml=youtube_manager.vectordata_yaml,
        chunk_size_seconds=chunk_size_seconds
    )
    
    return channel_manager.add_channel_to_vectorstore(
        channel_handle=channel_handle,
        youtube_api_key=youtube_api_key,
        season_start=season_start,
        season_end=season_end,
        chunk_size_seconds=chunk_size_seconds,
        skip_existing=skip_existing,
        estimate_only=estimate_only
    )


def _estimate_processing_requirements(
    videos: List[Dict[str, any]],
    chunk_size_seconds: int = 30
) -> Dict[str, float]:
    """
    Estimate time and memory requirements for processing YouTube videos.
    
    This is a backward compatibility wrapper. For new code, prefer using
    YoutubeChannelVectorManager._estimate_processing_requirements() directly.
    
    Args:
        videos: List of video dictionaries with duration and other info
        chunk_size_seconds: Size of transcript chunks in seconds
        
    Returns:
        Dictionary with estimated_time_minutes and estimated_memory_mb
    """
    # Create a minimal manager instance just for estimation
    manager = YoutubeChannelVectorManager(
        vectorstore=None,
        embeddings=None,  # Not needed for estimation
        persist_directory="",  # Not needed for estimation
        chunk_size_seconds=chunk_size_seconds
    )
    return manager._estimate_processing_requirements(videos, chunk_size_seconds)