# YouTube Channel Podcast Pipeline

This document describes the pipeline for automatically fetching all videos from a YouTube channel published during the NBA season and adding them to the local ChromaDB vectorstore.

## Overview

The pipeline:
1. Fetches all videos from a YouTube channel using the YouTube Data API v3
2. Filters videos by publish date (NBA season: July 2025 - June 2026)
3. Downloads transcripts for each video
4. Chunks transcripts into 30-second segments
5. Generates embeddings and adds to ChromaDB vectorstore

## Setup

### 1. Install Dependencies

```bash
pip install google-api-python-client
```

### 2. Get YouTube Data API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the YouTube Data API v3
4. Create credentials (API Key)
5. Add the key to your `.env` file:

```bash
YOUTUBE_API_KEY=your_api_key_here
```

### 3. Usage

#### Using VectorstoreManager (Recommended)

```python
from ClutchAI.rag.vector_manager import VectorstoreManager
import os

# Initialize manager
manager = VectorstoreManager(
    openai_api_key=os.environ.get('OPENAI_API_KEY'),
)

# Add channel podcasts
results = manager.add_youtube_channel_podcasts(
    channel_handle="@LockedOnFantasyBasketball",
    youtube_api_key=os.environ.get('YOUTUBE_API_KEY'),
    season_start="2025-07-01",
    season_end="2026-06-30",
    chunk_size_seconds=30,
    skip_existing=True
)

print(f"Videos added: {results['videos_added']}")
print(f"Chunks added: {results['chunks_added']}")
```

#### Using Example Script

```bash
python scripts/add_youtube_channel_podcasts.py
```

#### Get Estimates Only

To see time and memory estimates without processing:

```python
estimates = manager.add_youtube_channel_podcasts(
    channel_handle="@LockedOnFantasyBasketball",
    youtube_api_key=os.environ.get('YOUTUBE_API_KEY'),
    season_start="2025-07-01",
    season_end="2026-06-30",
    estimate_only=True
)

print(f"Estimated time: {estimates['estimated_time_minutes']:.1f} minutes")
print(f"Estimated memory: {estimates['estimated_memory_mb']:.1f} MB")
```

## Time and Memory Estimates

### Estimation Methodology

The pipeline estimates time and memory based on:

1. **Number of videos**: Fetched from YouTube API
2. **Average video duration**: Extracted from video metadata
3. **Chunk size**: Default 30 seconds per chunk
4. **Processing operations**:
   - YouTube API fetch: ~0.5s per video
   - Transcript download: ~2-5s per video
   - Text chunking: ~0.1s per chunk
   - Embedding generation: ~0.5s per chunk (OpenAI API)
   - Vectorstore insertion: ~0.1s per chunk

### Example Estimates

For a typical NBA season (July 2025 - June 2026) with the Locked On Fantasy Basketball channel:

**Assumptions:**
- ~250-300 videos (daily podcast, ~5-6 per week)
- Average video length: 30-45 minutes
- Chunk size: 30 seconds

**Estimated Results:**
- **Total videos**: ~250-300
- **Total chunks**: ~15,000-27,000 chunks
- **Processing time**: ~2-4 hours
- **Memory usage**: ~100-200 MB

### Detailed Breakdown

#### Time Estimates

- **API Fetch**: 250 videos × 0.5s = ~2 minutes
- **Transcript Download**: 250 videos × 3.5s = ~15 minutes
- **Chunking**: 20,000 chunks × 0.1s = ~33 minutes
- **Embedding**: 20,000 chunks × 0.5s = ~167 minutes (2.8 hours)
- **Insertion**: 20,000 chunks × 0.1s = ~33 minutes
- **Total**: ~4.5 hours (with 20% overhead)

*Note: Actual time may vary based on network speed, API rate limits, and video availability.*

#### Memory Estimates

- **Per chunk**: ~7 KB (text + embedding)
- **Total data**: 20,000 chunks × 7 KB = ~140 MB
- **With overhead**: ~210 MB (50% for ChromaDB and Python objects)

*Note: ChromaDB stores data on disk, so peak memory usage is lower than total storage.*

## Parameters

### `channel_handle`
- **Type**: `str`
- **Description**: YouTube channel handle (e.g., `@LockedOnFantasyBasketball`) or channel ID
- **Examples**: 
  - `"@LockedOnFantasyBasketball"`
  - `"UCxxxxxxxxxxxxxxxxxxxxx"` (channel ID)
  - `"https://www.youtube.com/@LockedOnFantasyBasketball"` (full URL)

### `season_start` / `season_end`
- **Type**: `str`
- **Format**: `YYYY-MM-DD`
- **Default**: `"2025-07-01"` / `"2026-06-30"`
- **Description**: Date range for filtering videos

### `chunk_size_seconds`
- **Type**: `int`
- **Default**: `30`
- **Description**: Size of transcript chunks in seconds
- **Note**: Smaller chunks = more granular retrieval but more documents

### `skip_existing`
- **Type**: `bool`
- **Default**: `True`
- **Description**: Skip videos already in vectorstore (based on video ID)

### `estimate_only`
- **Type**: `bool`
- **Default**: `False`
- **Description**: If `True`, only return estimates without processing videos

## Output

The pipeline returns a dictionary with:

```python
{
    'videos_found': 250,           # Total videos found in date range
    'videos_added': 245,            # Videos successfully added
    'videos_skipped': 5,            # Videos already in vectorstore
    'videos_failed': 0,             # Videos that failed (no transcript, etc.)
    'chunks_added': 20000,          # Total document chunks added
    'estimated_time_minutes': 270,  # Estimated processing time
    'estimated_memory_mb': 210      # Estimated memory usage
}
```

## Limitations

1. **Transcript Availability**: Videos must have captions/transcripts. Videos without transcripts will fail.
2. **API Rate Limits**: YouTube Data API has daily quotas (default: 10,000 units/day). Each video fetch uses ~100 units.
3. **OpenAI API Rate Limits**: Embedding generation is rate-limited. Large batches may take longer.
4. **Network Speed**: Transcript download speed depends on network connection.

## Troubleshooting

### Channel Not Found
- Verify the channel handle is correct
- Try using the full channel URL or channel ID instead
- Check that the YouTube Data API is enabled

### No Transcripts Available
- Some videos may not have captions
- Private or unlisted videos may not be accessible
- The pipeline will skip these videos and report them as failed

### API Quota Exceeded
- YouTube Data API has daily quotas
- Wait 24 hours or request a quota increase
- Process videos in smaller batches using `max_results` parameter

## See Also

- [Vectorstore Management Guide](VECTORSTORE_MANAGEMENT.md)
- [ChromaDB Setup Guide](CHROMADB_SETUP.md)


