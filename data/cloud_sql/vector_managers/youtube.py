"""
YouTube vector managers for ingesting YouTube videos into a PostgreSQL vectorstore.

This module provides managers for YouTube-specific operations including:
- Loading videos from YAML configuration
- Loading video transcripts
- Adding videos to vectorstore
- Fetching entire YouTube channels
"""

from __future__ import annotations

import os
import yaml
import re
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from urllib.parse import urlparse

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.youtube import TranscriptFormat

from data.cloud_sql.connection import PostgresConnection
from data.cloud_sql.vector_managers.base import BaseVectorManager
from data.data_class import YouTubeVideo, extract_youtube_video_id
from logger import get_logger

logger = get_logger(__name__)


class YoutubeVectorManager(BaseVectorManager):
    """
    Manager for ingesting YouTube videos into a PostgreSQL vectorstore.
    
    This class handles all YouTube-specific operations including:
    - Loading videos from YAML configuration
    - Loading video transcripts
    - Adding videos to vectorstore
    - Deleting videos from vectorstore
    - Handling force updates
    """
    
    def __init__(
        self,
        connection: PostgresConnection,
        embeddings: OpenAIEmbeddings,
        table_name: Optional[str] = None,
        vectordata_yaml: Optional[Path] = None,
        chunk_size_seconds: int = 30,
        max_retries: int = 3,
        retry_delay: float = 10.0,
    ):
        """
        Initialize the YoutubeVectorManager.
        
        Args:
            connection: PostgresConnection instance
            embeddings: OpenAIEmbeddings instance for document embedding
            table_name: Name of the vector table in PostgreSQL
            vectordata_yaml: Path to YAML file with video data (optional, can be passed to methods)
            chunk_size_seconds: Size of transcript chunks in seconds (default: 30)
            max_retries: Maximum number of retry attempts for IP blocking errors (default: 3)
            retry_delay: Initial delay in seconds between retries, doubles with each retry (default: 10.0)
        """
        super().__init__(connection, embeddings, table_name, vectordata_yaml)
        self.chunk_size_seconds = chunk_size_seconds
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def _fetch_video_metadata_from_api(
        self, 
        video_id: str, 
        youtube_api_key: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch video metadata from YouTube Data API v3.
        
        Args:
            video_id: YouTube video ID
            youtube_api_key: YouTube Data API v3 key (optional, will try GOOGLE_CLOUD_KEY env var if not provided)
            
        Returns:
            Dictionary with 'title', 'publish_date', 'description', etc., or None if unavailable
        """
        try:
            # Try to get API key
            try:
                youtube_api_key = self.get_env_var(
                    env_var_name='GOOGLE_CLOUD_KEY',
                    param_value=youtube_api_key,
                    help_url='https://console.cloud.google.com/apis/credentials',
                    description='Google Cloud API key (for YouTube Data API v3)'
                )
            except ValueError:
                # API key not available, skip metadata fetching
                return None
            
            from googleapiclient.discovery import build
            from googleapiclient.errors import HttpError
            
            youtube = build('youtube', 'v3', developerKey=youtube_api_key)
            
            # Fetch video details
            video_response = youtube.videos().list(
                part='snippet',
                id=video_id
            ).execute()
            
            if not video_response.get('items'):
                return None
            
            video = video_response['items'][0]
            snippet = video['snippet']
            
            # Extract publish date and format it
            published_at = snippet.get('publishedAt', '')
            publish_date = None
            if published_at:
                try:
                    # Parse ISO 8601 format: 2024-01-15T10:30:00Z
                    dt = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                    publish_date = dt.strftime('%Y-%m-%d')
                except Exception:
                    pass
            
            return {
                'title': snippet.get('title', ''),
                'description': snippet.get('description', ''),
                'publish_date': publish_date,
                'channel_title': snippet.get('channelTitle', ''),
                'tags': snippet.get('tags', [])
            }
        except ImportError:
            # google-api-python-client not installed
            return None
        except (HttpError, ValueError, Exception) as e:
            # API error or other issue - fail silently and return None
            return None
    
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
            logger.warning(f"YAML file not found at {yaml_path}")
            return []
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                
                if not data:
                    logger.warning("YAML file is empty.")
                    return []
                
                videos = []
                # Iterate through all top-level keys, but only process 'youtube' key
                for source_type, items in data.items():
                    if source_type != 'youtube' or not isinstance(items, list):
                        continue
                    
                    for item_data in items:
                        if not isinstance(item_data, dict):
                            logger.warning(f"Skipping invalid entry in {source_type}: {item_data}")
                            continue
                        
                        # Ensure required fields
                        if 'title' not in item_data or 'url' not in item_data:
                            logger.warning(f"Skipping {source_type} entry without title or url: {item_data}")
                            continue
                        
                        video = YouTubeVideo.from_dict(item_data)
                        videos.append((source_type, video))
                
                return videos
        except (yaml.YAMLError, KeyError, TypeError) as e:
            logger.warning(f"Error loading YAML file: {e}. Starting with empty list.")
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
        # Normalize URL before loading (remove timestamp parameters that can cause errors)
        url_normalized = self.normalize_url(url)
        
        # Use instance default if not provided
        chunk_size = chunk_size_seconds if chunk_size_seconds is not None else self.chunk_size_seconds
        
        # Retry logic with exponential backoff for IP blocking errors
        # Option 1: Wait and retry - YouTube IP blocks are often temporary
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                loader = YoutubeLoader.from_youtube_url(
                    url_normalized,
                    add_video_info=False,  # Always False - never works reliably, use YouTube Data API instead
                    transcript_format=TranscriptFormat.CHUNKS,
                    chunk_size_seconds=chunk_size
                )
                docs = loader.load()
                break  # Success, exit retry loop
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()
                
                # Check if it's an IP blocking error
                is_ip_block = (
                    'blocking' in error_msg or 
                    'blocked' in error_msg or 
                    'ip' in error_msg or
                    'too many requests' in error_msg
                )
                
                if attempt < self.max_retries and is_ip_block:
                    # Exponential backoff: delay doubles with each retry
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"  ⚠️  YouTube IP blocking detected. Waiting {delay:.1f} seconds before retry... (attempt {attempt + 1}/{self.max_retries + 1})")
                    time.sleep(delay)
                    continue
                else:
                    # Not a retryable error or max retries reached
                    raise ValueError(f"Failed to load transcript from {url}: {e}") from e
        
        # If we get here, docs should be loaded successfully
        # Auto-extract resource_id if not provided
        if not resource_id:
            resource_id = extract_youtube_video_id(url)
        
        # Fetch metadata from YouTube Data API if missing
        api_metadata = None
        if resource_id and (not title or not publish_date):
            # Only fetch if we're missing some metadata
            api_metadata = self._fetch_video_metadata_from_api(resource_id)
        
        # Extract video metadata from API response or document metadata
        video_publish_date = None
        video_title_from_info = None
        
        # Use API metadata if available
        if api_metadata:
            if not title and api_metadata.get('title'):
                video_title_from_info = api_metadata.get('title')
            if not publish_date and api_metadata.get('publish_date'):
                video_publish_date = api_metadata.get('publish_date')
        
        # Fallback to document metadata if still missing (though unlikely with add_video_info=False)
        if docs and docs[0].metadata:
            if not video_title_from_info and 'title' in docs[0].metadata:
                video_title_from_info = docs[0].metadata.get('title')
            if not video_publish_date and 'publish_date' in docs[0].metadata and not publish_date:
                try:
                    # Parse the publish_date from video info (format: YYYY-MM-DD)
                    video_publish_date_str = docs[0].metadata.get('publish_date')
                    if video_publish_date_str:
                        # Handle different date formats
                        if isinstance(video_publish_date_str, str):
                            # Try parsing common formats
                            for fmt in ['%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%SZ']:
                                try:
                                    dt = datetime.strptime(video_publish_date_str.split('T')[0], '%Y-%m-%d')
                                    video_publish_date = dt.strftime('%Y-%m-%d')
                                    break
                                except:
                                    continue
                except Exception:
                    pass  # If parsing fails, use provided dates or None
        
        # Use extracted publish_date if not provided
        final_publish_date = publish_date or video_publish_date
        # Use upload_date as fallback for publish_date if neither provided
        final_upload_date = upload_date or video_publish_date
        final_title = title or video_title_from_info
        
        # Enhance metadata for all documents
        for doc in docs:
                # Ensure metadata dict exists
                if not doc.metadata:
                    doc.metadata = {}
                
                # Add/update custom metadata fields
                doc.metadata['source_type'] = source_type
                doc.metadata['url'] = url_normalized
                
                # Add resource_id if available
                if resource_id:
                    doc.metadata['resource_id'] = resource_id
                
                # Add title if available (prefer provided, then extracted)
                if final_title:
                    doc.metadata['title'] = final_title
                
                # Add upload_date if available
                if final_upload_date:
                    doc.metadata['upload_date'] = self._ensure_string_date(final_upload_date)
                
                # Add publish_date if available (defaults to upload_date if not provided)
                if final_publish_date:
                    doc.metadata['publish_date'] = self._ensure_string_date(final_publish_date)
                elif final_upload_date:
                    doc.metadata['publish_date'] = self._ensure_string_date(final_upload_date)
                
            # Keep existing metadata from YoutubeLoader:
            # - 'source': Full URL with timestamp
            # - 'start_seconds': Start time of the chunk in seconds
            # - 'start_timestamp': Human-readable timestamp
        
        return docs
    
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
            logger.warning(f"No transcript found for {url}")
            return 0
        
        # Add documents to vectorstore using base class method
        self._add_documents_to_vectorstore(docs)
        
        return len(docs)
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds into HH:MM:SS timestamp."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
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
    Manager for ingesting entire YouTube channels into a PostgreSQL vectorstore.
    
    This class extends YoutubeVectorManager with channel-level operations including:
    - Fetching all videos from a YouTube channel
    - Processing channel videos in bulk
    - Estimating processing requirements
    - Adding channel videos to vectorstore
    """
    
    def fetch_channel_videos(
        self,
        channel_handle: str,
        youtube_api_key: Optional[str] = None,
        published_after: Optional[str] = None,
        published_before: Optional[str] = None,
        max_results: Optional[int] = None
    ) -> List[Dict[str, any]]:
        """
        Fetch all videos from a YouTube channel using YouTube Data API v3.
        
        Args:
            channel_handle: YouTube channel handle (e.g., '@LockedOnFantasyBasketball' or 'UC...')
            youtube_api_key: YouTube Data API v3 key (optional, will try GOOGLE_CLOUD_KEY env var if not provided)
            published_after: Filter videos published after this date (ISO 8601 format: YYYY-MM-DD)
            published_before: Filter videos published before this date (ISO 8601 format: YYYY-MM-DD)
            max_results: Maximum number of videos to fetch (None for all)
            
        Returns:
            List of dictionaries with video information (id, title, url, publishedAt, etc.)
            
        Raises:
            ValueError: If YouTube API key is not provided and GOOGLE_CLOUD_KEY environment variable is not set
        """
        # Get API key using base class utility
        # Note: We don't store this as instance variable - only check when needed for security
        youtube_api_key = self.get_env_var(
            env_var_name='GOOGLE_CLOUD_KEY',
            param_value=youtube_api_key,
            help_url='https://console.cloud.google.com/apis/credentials',
            description='Google Cloud API key (for YouTube Data API v3)'
        )
        
        try:
            from googleapiclient.discovery import build
            from googleapiclient.errors import HttpError
        except ImportError:
            raise ImportError(
                "google-api-python-client is required. Install it with: pip install google-api-python-client"
            )
        
        youtube = build('youtube', 'v3', developerKey=youtube_api_key)
        videos = []
        
        # Convert channel handle to channel ID if needed
        channel_id = None
        if channel_handle.startswith('@'):
            handle_name = channel_handle[1:]
            try:
                search_response = youtube.search().list(
                    q=channel_handle,
                    type='channel',
                    part='id',
                    maxResults=1
                ).execute()
                
                if search_response.get('items'):
                    channel_id = search_response['items'][0]['id']['channelId']
                else:
                    raise ValueError(f"Channel not found: {channel_handle}")
            except HttpError as e:
                raise ValueError(f"Error finding channel {channel_handle}: {e}")
        elif channel_handle.startswith('UC') and len(channel_handle) == 24:
            channel_id = channel_handle
        else:
            if '/' in channel_handle:
                match = re.search(r'@([^/?]+)', channel_handle)
                if match:
                    handle_name = match.group(1)
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
                channel_id = channel_handle
        
        # Get the channel's uploads playlist ID (recommended approach for getting all videos)
        try:
            channel_response = youtube.channels().list(
                part='contentDetails',
                id=channel_id
            ).execute()
            
            if not channel_response.get('items'):
                raise ValueError(f"Channel not found: {channel_handle}")
            
            uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        except HttpError as e:
            raise ValueError(f"YouTube API error getting channel details: {e}")
        
        # Convert date strings to datetime for comparison
        published_after_dt = None
        published_before_dt = None
        
        if published_after:
            published_after_dt = datetime.fromisoformat(f"{published_after}T00:00:00")
        if published_before:
            published_before_dt = datetime.fromisoformat(f"{published_before}T23:59:59")
        
        # Fetch all videos from the uploads playlist using playlistItems API
        # This is more reliable than the Search API for getting all videos
        next_page_token = None
        video_ids = []
        
        while True:
            playlist_params = {
                'playlistId': uploads_playlist_id,
                'part': 'contentDetails',
                'maxResults': 50
            }
            
            if next_page_token:
                playlist_params['pageToken'] = next_page_token
            
            try:
                playlist_response = youtube.playlistItems().list(**playlist_params).execute()
            except HttpError as e:
                raise ValueError(f"YouTube API error fetching playlist items: {e}")
            
            # Collect video IDs from this page
            for item in playlist_response.get('items', []):
                video_id = item['contentDetails']['videoId']
                video_ids.append(video_id)
            
            # Check for next page
            next_page_token = playlist_response.get('nextPageToken')
            if not next_page_token:
                break
        
        # Get detailed video information in batches (API limit is 50 per request)
        # Filter by date range in Python code for reliability
        batch_size = 50
        for i in range(0, len(video_ids), batch_size):
            batch_ids = video_ids[i:i + batch_size]
            
            try:
                video_details = youtube.videos().list(
                    part='snippet,contentDetails,statistics',
                    id=','.join(batch_ids)
                ).execute()
            except HttpError as e:
                raise ValueError(f"YouTube API error getting video details: {e}")
            
            for video in video_details.get('items', []):
                published_at_str = video['snippet']['publishedAt']
                # Parse ISO format datetime (e.g., "2025-12-17T15:00:00Z")
                # YouTube API returns ISO 8601 format with 'Z' for UTC
                # Use regex to extract just the date-time part (YYYY-MM-DDTHH:MM:SS)
                # Remove timezone indicators (Z, +HH:MM, -HH:MM)
                match = re.match(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', published_at_str)
                if match:
                    published_at_clean = match.group(1)
                else:
                    # Fallback: just remove Z and timezone offsets
                    published_at_clean = published_at_str.replace('Z', '').split('+')[0]
                    if published_at_clean.count('-') > 2:
                        # Has negative timezone, extract datetime part
                        parts = published_at_clean.rsplit('-', 2)
                        if len(parts) > 2 and ':' in parts[-1]:
                            published_at_clean = '-'.join(parts[:-2])
                
                published_at_dt = datetime.fromisoformat(published_at_clean)
                
                # Apply date range filter
                if published_after_dt and published_at_dt < published_after_dt:
                    continue
                if published_before_dt and published_at_dt > published_before_dt:
                    continue
                
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
        
        return videos
    
    def _get_database_storage_info(self) -> Optional[Dict[str, Any]]:
        """
        Get database storage information including database size and video count.
        
        Returns:
            Dictionary with storage information or None if error
        """
        try:
            from sqlalchemy import text
            engine = self.connection.get_engine()
            
            with engine.connect() as conn:
                # Get database name
                db_name = self.connection.vectordb
                
                # Get database size
                result = conn.execute(text("""
                    SELECT pg_size_pretty(pg_database_size(:db_name)) as db_size,
                           pg_database_size(:db_name) as db_size_bytes
                """), {"db_name": db_name})
                db_row = result.fetchone()
                database_size = db_row[0] if db_row else "N/A"
                database_size_bytes = db_row[1] if db_row and len(db_row) > 1 else 0
                
                # Get collection UUID for this table
                table_name = self.table_name
                result = conn.execute(text("""
                    SELECT uuid FROM langchain_pg_collection 
                    WHERE name = :collection_name
                """), {"collection_name": table_name})
                collection_row = result.fetchone()
                
                video_count = 0
                if collection_row:
                    collection_uuid = collection_row[0]
                    
                    # Count distinct videos (resources with source_type='youtube')
                    result = conn.execute(text("""
                        SELECT COUNT(DISTINCT (cmetadata->>'resource_id')) 
                        FROM langchain_pg_embedding 
                        WHERE collection_id = :collection_uuid
                        AND cmetadata->>'source_type' = 'youtube'
                        AND cmetadata->>'resource_id' IS NOT NULL
                    """), {"collection_uuid": collection_uuid})
                    video_count = result.scalar() or 0
                
                return {
                    'database_size': database_size,
                    'database_size_bytes': database_size_bytes,
                    'video_count': video_count,
                }
        except Exception as e:
            # Return None on error, caller will handle it
            return None
    
    @staticmethod
    def _format_bytes(bytes_value: int) -> str:
        """Format bytes to human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.2f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.2f} PB"
    
    def _estimate_processing_requirements(
        self,
        videos: List[Dict[str, any]],
        chunk_size_seconds: Optional[int] = None,
        skip_existing: bool = True,
        delay_between_videos: float = 3.0,
        estimated_success_rate: float = 0.15,
        max_videos_added: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Estimate time and memory requirements for processing YouTube videos.
        
        Based on actual performance data:
        - Average ~63 chunks per successful video
        - ~15% success rate (accounting for IP blocking, missing transcripts, etc.)
        - ~5-10 seconds per successful video processing
        
        Args:
            videos: List of video dictionaries with duration and other info
            chunk_size_seconds: Size of transcript chunks in seconds (uses instance default if not provided)
            skip_existing: Whether to skip existing videos (affects estimate)
            delay_between_videos: Delay in seconds between videos (affects total time)
            estimated_success_rate: Estimated percentage of videos that will succeed (default: 0.15 = 15%)
            max_videos_added: Maximum number of videos to add (stops early, None for all)
            
        Returns:
            Dictionary with estimated_time_minutes and estimated_memory_mb
        """
        chunk_size = chunk_size_seconds if chunk_size_seconds is not None else self.chunk_size_seconds
        
        total_videos = len(videos)
        
        # If max_videos_added is set, estimate how many videos we'll need to process
        # to get that many successful additions
        if max_videos_added is not None:
            # Estimate: we need to process enough videos to get max_videos_added successful ones
            # Account for skip_existing: if skipping, we need to process more videos
            if skip_existing:
                # Assume some videos will be skipped, so we need to process more
                # Estimate: if success_rate is 15%, and 5% are skipped, we need:
                # max_videos_added / (success_rate * (1 - skip_rate))
                skip_rate = 0.05  # Assume 5% will be skipped
                effective_success_rate = estimated_success_rate * (1 - skip_rate)
                videos_to_process = max_videos_added / effective_success_rate if effective_success_rate > 0 else total_videos
                # Cap at total videos available
                videos_to_process = min(videos_to_process, total_videos)
            else:
                # No skipping, so we need max_videos_added / success_rate
                videos_to_process = max_videos_added / estimated_success_rate if estimated_success_rate > 0 else total_videos
                videos_to_process = min(videos_to_process, total_videos)
            
            # Update estimated successful videos to match max_videos_added
            estimated_successful = max_videos_added
        else:
            # Estimate videos that will actually be processed
            # Based on actual data: ~15% success rate, some skipped
            if skip_existing:
                # Assume ~1-5% will be skipped (already in DB)
                videos_to_process = total_videos * 0.95
            else:
                videos_to_process = total_videos
            
            # Estimate successful videos (based on actual: 14/175 = 8%, but using 15% as conservative)
            estimated_successful = videos_to_process * estimated_success_rate
        
        # Estimate successful videos (based on actual: 14/175 = 8%, but using 15% as conservative)
        estimated_successful = videos_to_process * estimated_success_rate
        
        # Calculate chunks based on actual average (~63 chunks per video)
        # This is more accurate than calculating from duration
        avg_chunks_per_video = 63  # Based on actual: 879 chunks / 14 videos
        estimated_total_chunks = int(estimated_successful * avg_chunks_per_video)
        
        # Time estimates based on actual performance
        # Per video processing time (including retries, failures, etc.)
        time_per_video_attempt = 2.0  # Average time per video attempt (including failures)
        retry_overhead = 1.2  # Account for retry delays (exponential backoff)
        
        # Total processing time
        processing_time = videos_to_process * time_per_video_attempt * retry_overhead
        
        # Add delays between videos (critical factor!)
        delay_time = videos_to_process * delay_between_videos
        
        # Embedding and insertion time (batch operations, much faster than per-chunk)
        # Based on actual: embeddings are fast, insertion is batched
        embedding_insertion_time = estimated_total_chunks * 0.02  # ~20ms per chunk (batched)
        
        total_time_seconds = processing_time + delay_time + embedding_insertion_time
        estimated_time_minutes = total_time_seconds / 60
        
        # Memory estimates (more realistic)
        # Each chunk: text (~1KB) + embedding vector (~1.5KB) + metadata (~0.5KB) = ~3KB
        bytes_per_chunk = 3 * 1024  # 3 KB per chunk
        total_data_mb = (estimated_total_chunks * bytes_per_chunk) / (1024 * 1024)
        # Add overhead for processing
        estimated_memory_mb = total_data_mb * 2.0
        
        return {
            'estimated_time_minutes': estimated_time_minutes,
            'estimated_memory_mb': estimated_memory_mb,
            'estimated_chunks': estimated_total_chunks,
            'estimated_successful_videos': int(estimated_successful),
            'estimated_total_duration_hours': 0  # Not calculated from video duration anymore
        }
    
    def add_channel_to_vectorstore(
        self,
        channel_handle: str,
        youtube_api_key: Optional[str] = None,
        season_start: str = "2025-07-01",
        season_end: str = "2026-06-30",
        chunk_size_seconds: Optional[int] = None,
        skip_existing: bool = True,
        estimate_only: bool = False,
        delay_between_videos: float = 2.0,
        max_videos: Optional[int] = None,
        max_videos_added: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Pipeline to fetch all videos from a YouTube channel published during NBA season and add to vectorstore.
        
        Args:
            channel_handle: YouTube channel handle (e.g., '@LockedOnFantasyBasketball')
            youtube_api_key: YouTube Data API v3 key (optional, will try GOOGLE_CLOUD_KEY env var if not provided)
            season_start: Start date of NBA season (YYYY-MM-DD format, default: 2025-07-01)
            season_end: End date of NBA season (YYYY-MM-DD format, default: 2026-06-30)
            chunk_size_seconds: Size of transcript chunks in seconds (uses instance default if not provided)
            skip_existing: Skip videos already in vectorstore
            estimate_only: If True, only return estimates without processing videos
            delay_between_videos: Delay in seconds between processing videos to avoid rate limiting (default: 2.0)
            max_videos: Maximum number of videos to process (processes most recent N videos, None for all)
            max_videos_added: Maximum number of videos to add (stops after N videos are successfully added, None for all)
            
        Returns:
            Dictionary with results including videos_found, videos_added, chunks_added, etc.
            
        Raises:
            ValueError: If YouTube API key is not provided and GOOGLE_CLOUD_KEY environment variable is not set
        """
        # Get API key using base class utility
        # Note: We don't store this as instance variable - only check when needed for security
        youtube_api_key = self.get_env_var(
            env_var_name='GOOGLE_CLOUD_KEY',
            param_value=youtube_api_key,
            help_url='https://console.cloud.google.com/apis/credentials',
            description='Google Cloud API key (for YouTube Data API v3)'
        )
        chunk_size = chunk_size_seconds if chunk_size_seconds is not None else self.chunk_size_seconds
        
        logger.info(f"Fetching videos from channel: {channel_handle}")
        logger.info(f"Season range: {season_start} to {season_end}")
        
        # Fetch videos from channel
        videos = self.fetch_channel_videos(
            channel_handle=channel_handle,
            youtube_api_key=youtube_api_key,
            published_after=season_start,
            published_before=season_end
        )
        
        total_videos_found = len(videos)
        logger.info(f"Found {total_videos_found} videos in season range")
        
        # Limit to most recent N videos if max_videos is specified
        # Videos are already sorted by date (newest first) from the API
        if max_videos is not None and max_videos > 0:
            videos = videos[:max_videos]
            logger.info(f"Limiting to most recent {len(videos)} videos (max_videos={max_videos})")
        
        videos_to_process = len(videos)
        
        if estimate_only:
            return self._estimate_processing_requirements(
                videos, 
                chunk_size, 
                skip_existing=skip_existing,
                delay_between_videos=delay_between_videos
            )
        
        # Get existing resource IDs if skipping
        existing_resource_ids = set()
        if skip_existing:
            existing_resource_ids = self.get_existing_resource_ids()
        
        results = {
            'videos_found': total_videos_found,  # Total videos found in season range
            'videos_to_process': videos_to_process,  # Videos actually being processed (after max_videos limit)
            'videos_added': 0,
            'videos_skipped': 0,
            'videos_failed': 0,
            'chunks_added': 0,
        }
        
        logger.info(f"\nProcessing videos...")
        
        # Track actual processing time
        processing_start_time = time.time()
        
        # Process each video
        for i, video in enumerate(videos, 1):
            video_id = video['id']
            video_url = video['url']
            video_title = video['title']
            published_at = video['publishedAt']
            
            # Parse publish date
            publish_date = datetime.fromisoformat(published_at.replace('Z', '+00:00')).strftime('%Y-%m-%d')
            
            logger.info(f"\n[{i}/{len(videos)}] Processing: {video_title}")
            
            # Check if already exists
            if skip_existing and video_id in existing_resource_ids:
                logger.info(f"  ⏭ Skipping (already in vectorstore)")
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
                    logger.info(f"  ✓ Added {chunks_added} chunks")
                    
                    # Check if we've reached the limit for videos added
                    if max_videos_added is not None and results['videos_added'] >= max_videos_added:
                        logger.info(f"\n  🛑 Reached limit of {max_videos_added} videos added. Stopping.")
                        break
                else:
                    results['videos_failed'] += 1
                    logger.warning(f"  ✗ Failed (no transcript found)")
            except Exception as e:
                results['videos_failed'] += 1
                error_msg = str(e).lower()
                if 'blocking' in error_msg or 'blocked' in error_msg:
                    logger.error(f"  ✗ Failed: YouTube IP blocking (consider using a proxy)")
                else:
                    logger.error(f"  ✗ Failed: {e}")
            
            # Add delay between videos to avoid rate limiting (except after last video or if we've reached the limit)
            if i < len(videos) and delay_between_videos > 0:
                # Don't delay if we've reached the max_videos_added limit
                if max_videos_added is None or results['videos_added'] < max_videos_added:
                    time.sleep(delay_between_videos)
        
        # Calculate actual processing time
        processing_end_time = time.time()
        actual_time_seconds = processing_end_time - processing_start_time
        actual_time_minutes = actual_time_seconds / 60.0
        results['actual_time_seconds'] = actual_time_seconds
        results['actual_time_minutes'] = actual_time_minutes
        
        # Format duration for display
        hours = int(actual_time_minutes // 60)
        minutes = int(actual_time_minutes % 60)
        seconds = int(actual_time_seconds % 60)
        if hours > 0:
            duration_str = f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            duration_str = f"{minutes}:{seconds:02d}"
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing complete!")
        logger.info(f"  Videos found in season: {results['videos_found']}")
        if results['videos_found'] != results.get('videos_to_process', results['videos_found']):
            logger.info(f"  Videos processed: {results.get('videos_to_process', results['videos_found'])} (limited by max_videos)")
        logger.info(f"  Videos added: {results['videos_added']}")
        logger.info(f"  Videos skipped: {results['videos_skipped']}")
        logger.info(f"  Videos failed: {results['videos_failed']}")
        logger.info(f"  Total chunks added: {results['chunks_added']}")
        logger.info(f"  Actual time: {duration_str} ({actual_time_minutes:.1f} minutes)")
        logger.info(f"{'='*60}")
        
        # Get and display database storage information
        try:
            storage_info = self._get_database_storage_info()
            if storage_info:
                logger.info(f"\n{'='*60}")
                logger.info(f"Database Storage Summary")
                logger.info(f"{'='*60}")
                logger.info(f"  Database size (used): {storage_info.get('database_size', 'N/A')}")
                logger.info(f"  Total videos: {storage_info.get('video_count', 0)}")
                logger.info(f"{'='*60}")
        except Exception as e:
            logger.warning(f"\n⚠️  Could not retrieve database storage information: {e}")
        
        return results
