"""
Example script to add YouTube channel podcasts to vectorstore.

This script demonstrates how to use the pipeline to fetch all videos from
a YouTube channel published during the NBA season and add them to the vectorstore.

Usage:
    python scripts/add_youtube_channel_podcasts.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from ClutchAI.rag.vector_manager import VectorstoreManager

# Load environment variables
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

def main():
    # Initialize VectorstoreManager
    vectorstore_manager = VectorstoreManager(
        chroma_persist_directory=None,  # Uses default
        openai_api_key=os.environ.get('OPENAI_API_KEY'),
    )
    
    # Channel to process
    channel_handle = "@LockedOnFantasyBasketball"
    
    # NBA season dates (July 2025 to June 2026)
    season_start = "2025-07-01"
    season_end = "2026-06-30"
    
    print("=" * 60)
    print("YouTube Channel Podcast Pipeline")
    print("=" * 60)
    print(f"Channel: {channel_handle}")
    print(f"Season: {season_start} to {season_end}")
    print("=" * 60)
    
    # First, get estimates only
    print("\n📊 Getting estimates...")
    estimates = vectorstore_manager.add_youtube_channel_podcasts(
        channel_handle=channel_handle,
        youtube_api_key=os.environ.get('YOUTUBE_API_KEY'),
        season_start=season_start,
        season_end=season_end,
        chunk_size_seconds=30,
        estimate_only=True
    )
    
    print(f"\nEstimated videos: {estimates.get('videos_found', 0)}")
    print(f"Estimated chunks: {estimates.get('estimated_chunks', 0):,}")
    print(f"Estimated time: {estimates.get('estimated_time_minutes', 0):.1f} minutes")
    print(f"Estimated memory: {estimates.get('estimated_memory_mb', 0):.1f} MB")
    print(f"Estimated total duration: {estimates.get('estimated_total_duration_hours', 0):.1f} hours")
    
    # Ask for confirmation
    response = input("\nProceed with processing? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Cancelled.")
        return
    
    # Process videos
    print("\n🚀 Processing videos...")
    results = vectorstore_manager.add_youtube_channel_podcasts(
        channel_handle=channel_handle,
        youtube_api_key=os.environ.get('YOUTUBE_API_KEY'),
        season_start=season_start,
        season_end=season_end,
        chunk_size_seconds=30,
        skip_existing=True,
        estimate_only=False
    )
    
    print("\n✅ Complete!")
    print(f"Videos added: {results['videos_added']}")
    print(f"Videos skipped: {results['videos_skipped']}")
    print(f"Videos failed: {results['videos_failed']}")
    print(f"Total chunks: {results['chunks_added']:,}")

if __name__ == "__main__":
    main()

