from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional


def extract_youtube_video_id(url: str) -> Optional[str]:
    """
    Extract YouTube video ID from various URL formats.
    
    Supports:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - https://m.youtube.com/watch?v=VIDEO_ID
    
    Args:
        url: YouTube URL
        
    Returns:
        Video ID if found, None otherwise
    """
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com\/watch\?.*v=([a-zA-Z0-9_-]{11})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


@dataclass
class YouTubeVideo:
    """Data class for storing YouTube video information."""
    title: str
    url: str
    id: Optional[str] = None  # Unique identifier (auto-extracted for YouTube, custom for articles)
    added_at: Optional[str] = None
    description: Optional[str] = None
    upload_date: Optional[str] = None
    publish_date: Optional[str] = None
    force_update: bool = False
    
    def __post_init__(self):
        """Set added_at timestamp if not provided and extract ID if needed."""
        if self.added_at is None:
            self.added_at = datetime.now().isoformat()
        
        # Auto-extract YouTube video ID if not provided and URL is a YouTube URL
        if self.id is None and self.url:
            youtube_id = extract_youtube_video_id(self.url)
            if youtube_id:
                self.id = youtube_id
        
        # Convert force_update to bool if it's a string or other type
        if isinstance(self.force_update, str):
            self.force_update = self.force_update.lower() in ('true', '1', 'yes')
        elif not isinstance(self.force_update, bool):
            self.force_update = bool(self.force_update)
        
        # Convert date objects to strings (PyYAML may parse dates as date objects)
        if self.upload_date is not None and not isinstance(self.upload_date, str):
            if hasattr(self.upload_date, 'strftime'):
                self.upload_date = self.upload_date.strftime('%Y-%m-%d')
            else:
                self.upload_date = str(self.upload_date)
        
        if self.publish_date is not None and not isinstance(self.publish_date, str):
            if hasattr(self.publish_date, 'strftime'):
                self.publish_date = self.publish_date.strftime('%Y-%m-%d')
            else:
                self.publish_date = str(self.publish_date)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> YouTubeVideo:
        """Create YouTubeVideo from dictionary."""
        # Map resource_id from YAML to id field in dataclass
        if 'resource_id' in data and 'id' not in data:
            data['id'] = data.pop('resource_id')
        
        # Filter out None values for optional fields to avoid errors
        allowed_fields = ['title', 'url', 'id', 'added_at', 'description', 'upload_date', 'publish_date', 'force_update']
        filtered_data = {k: v for k, v in data.items() if k in allowed_fields}
        return cls(**filtered_data)

