"""
Script to update LockedOn Fantasy Basketball podcast episodes in vectorstore.

This script fetches all videos from the LockedOn Fantasy Basketball YouTube channel
published during the NBA season and adds them to the PostgreSQL vectorstore.

Usage:
    python scripts/vectordb_pipelines/update_lockedon_knowledge.py
"""

import os
import sys
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
# From scripts/vectordb_pipelines/update_lockedon_knowledge.py
# .parent = scripts/vectordb_pipelines/
# .parent.parent = scripts/
# .parent.parent.parent = project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data.cloud_sql.connection import PostgresConnection
from data.cloud_sql.vector_managers import YoutubeChannelVectorManager
from langchain_openai import OpenAIEmbeddings
from logger import get_logger, setup_logging

# Load environment variables
env_path = project_root / '.env'
load_dotenv(env_path)

# Setup logging
setup_logging(debug=False)
logger = get_logger(__name__)

# Load configuration from YAML files
# Load pipeline-specific config
vector_config_path = project_root / 'config' / 'vector_config.yaml'
vector_config = {}
if vector_config_path.exists():
    with open(vector_config_path, 'r') as f:
        vector_config = yaml.safe_load(f) or {}
else:
    logger.warning(f"Vector config file not found at {vector_config_path}, using defaults")

# Load RAG config for chunk_size_seconds (source of truth)
rag_config_path = project_root / 'config' / 'rag_config.yaml'
rag_config = {}
if rag_config_path.exists():
    with open(rag_config_path, 'r') as f:
        rag_config = yaml.safe_load(f) or {}
else:
    logger.warning(f"RAG config file not found at {rag_config_path}, using defaults")

def main():
    # Initialize PostgreSQL connection
    connection = PostgresConnection()
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(api_key=os.environ.get('OPENAI_API_KEY'))
    
    # Initialize YouTube channel manager
    youtube_manager = YoutubeChannelVectorManager(
        connection=connection,
        embeddings=embeddings,
        table_name=None,  # Uses CLOUDSQL_VECTOR_TABLE from env
        chunk_size_seconds=30,
    )
    
    # Channel to process
    channel_handle = "@LockedOnFantasyBasketball"
    
    # NBA season dates (July 2025 to June 2026)
    season_start = "2025-07-01"
    season_end = "2026-06-30"
    
    logger.info("=" * 60)
    logger.info("YouTube Channel Podcast Pipeline")
    logger.info("=" * 60)
    logger.info(f"Channel: {channel_handle}")
    logger.info(f"Season: {season_start} to {season_end}")
    logger.info("=" * 60)
    
    # Get configuration from YAML files
    # Pipeline-specific settings from vector_config.yaml
    youtube_config = vector_config.get('youtube_channel', {})
    max_videos_added = youtube_config.get('max_videos_added', 10)
    delay_between_videos = youtube_config.get('delay_between_videos', 12.0)
    
    # Chunk size from rag_config.yaml (source of truth for consistency)
    rag_youtube_config = rag_config.get('youtube', {})
    chunk_size_seconds = rag_youtube_config.get('chunk_size_seconds', 30)
    
    # Process videos
    logger.info("\n🚀 Processing videos...")
    logger.info("Note: Using delays between videos and retry logic to handle IP blocking")
    logger.info("      IP blocks are often temporary - the script will retry with exponential backoff")
    logger.info(f"      Will stop after {max_videos_added} videos are successfully added")
    results = youtube_manager.add_channel_to_vectorstore(
        channel_handle=channel_handle,
        season_start=season_start,
        season_end=season_end,
        chunk_size_seconds=chunk_size_seconds,
        skip_existing=True,
        estimate_only=False,
        delay_between_videos=delay_between_videos,
        max_videos=None,  # No limit on videos to process, only on videos added
        max_videos_added=max_videos_added
    )
    
    logger.info("\n✅ Complete!")
    logger.info(f"Videos added: {results['videos_added']}")
    logger.info(f"Videos skipped: {results['videos_skipped']}")
    logger.info(f"Videos failed: {results['videos_failed']}")
    logger.info(f"Total chunks: {results['chunks_added']:,}")

if __name__ == "__main__":
    main()
