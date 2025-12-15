"""
Example script to add YouTube channel podcasts to vectorstore.

This script demonstrates how to use the pipeline to fetch all videos from
a YouTube channel published during the NBA season and add them to the PostgreSQL vectorstore.

Usage:
    python scripts/add_youtube_channel_podcasts.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from data.cloud_sql.connection import PostgresConnection
from data.cloud_sql.vector_managers import YoutubeChannelVectorManager
from langchain_openai import OpenAIEmbeddings

# Load environment variables
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

def main():
    # Initialize PostgreSQL connection
    connection = PostgresConnection()
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(api_key=os.environ.get('OPENAI_API_KEY'))
    
    # Initialize YouTube channel manager
    youtube_manager = YoutubeChannelVectorManager(
        connection=connection,
        embeddings=embeddings,
        table_name="embeddings",
        chunk_size_seconds=30,
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
    
    # Process videos
    print("\n🚀 Processing videos...")
    results = youtube_manager.add_channel_to_vectorstore(
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

