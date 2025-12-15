"""
YouTube vector managers for ingesting YouTube videos into a PostgreSQL vectorstore.

This module provides managers for YouTube-specific operations including:
- Loading videos from YAML configuration
- Loading video transcripts
- Adding videos to vectorstore
- Fetching entire YouTube channels
"""

from __future__ import annotations

import yaml
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.youtube import TranscriptFormat

from data.cloud_sql.connection import PostgresConnection
from data.cloud_sql.vector_managers.base import BaseVectorManager
from data.data_class import YouTubeVideo, extract_youtube_video_id


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
        table_name: str = "embeddings",
        vectordata_yaml: Optional[Path] = None,
        chunk_size_seconds: int = 30,
    ):
        """
        Initialize the YoutubeVectorManager.
        
        Args:
            connection: PostgresConnection instance
            embeddings: OpenAIEmbeddings instance for document embedding
            table_name: Name of the vector table in PostgreSQL
            vectordata_yaml: Path to YAML file with video data (optional, can be passed to methods)
            chunk_size_seconds: Size of transcript chunks in seconds (default: 30)
        """
        super().__init__(connection, embeddings, table_name, vectordata_yaml)
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
                add_video_info=True,  # Enable to extract video metadata including publish date
                transcript_format=TranscriptFormat.CHUNKS,
                chunk_size_seconds=chunk_size
            )
            docs = loader.load()
            
            # Auto-extract resource_id if not provided
            if not resource_id:
                resource_id = extract_youtube_video_id(url)
            
            # Extract video metadata from first document if available
            video_publish_date = None
            video_title_from_info = None
            if docs and docs[0].metadata:
                # YoutubeLoader with add_video_info=True adds 'publish_date' to metadata
                if 'publish_date' in docs[0].metadata and not publish_date:
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
                
                # Extract title from video info if not provided
                if not title and 'title' in docs[0].metadata:
                    video_title_from_info = docs[0].metadata.get('title')
            
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
    
    # Backward compatibility methods
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
        
        # Build search parameters
        search_params = {
            'channelId': channel_id,
            'type': 'video',
            'part': 'id',
            'order': 'date',
            'maxResults': 50
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
        
        total_duration_seconds = 0
        total_videos = len(videos)
        
        for video in videos:
            duration_seconds = parse_duration(video.get('duration', ''))
            if duration_seconds == 0:
                duration_seconds = 30 * 60  # Default: 30 minutes
            total_duration_seconds += duration_seconds
        
        estimated_chunks_per_video = total_duration_seconds / chunk_size if total_videos > 0 else 0
        total_chunks = estimated_chunks_per_video * total_videos
        
        # Time estimates
        api_fetch_time = total_videos * 0.5
        transcript_time = total_videos * 3.5
        chunking_time = total_chunks * 0.1
        embedding_time = total_chunks * 0.5
        insertion_time = total_chunks * 0.1
        
        total_time_seconds = (api_fetch_time + transcript_time + chunking_time + 
                             embedding_time + insertion_time) * 1.2
        estimated_time_minutes = total_time_seconds / 60
        
        # Memory estimates
        bytes_per_chunk = 7 * 1024  # 7 KB
        total_data_mb = (total_chunks * bytes_per_chunk) / (1024 * 1024)
        estimated_memory_mb = total_data_mb * 1.5
        
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
            Dictionary with results including videos_found, videos_added, chunks_added, etc.
        """
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
