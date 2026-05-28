"""
One-time script to seed the lockedon_video_queue table.

Fetches all LockedOn Fantasy Basketball channel videos for the 2025-26 season,
inserts them into the queue, and marks already-ingested ones as 'done'.

Safe to re-run: ON CONFLICT DO NOTHING preserves existing rows.

Usage:
    python scripts/pipelines/seed_lockedon_queue.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from data.postgres.connection import PostgresConnection
from data.postgres.video_queue import VideoQueueManager
from data.postgres.vector_managers import YoutubeChannelVectorManager
from langchain_openai import OpenAIEmbeddings
from logger import get_logger, setup_logging

setup_logging(debug=False)
logger = get_logger(__name__)

CHANNEL = "@LockedOnFantasyBasketball"
SEASON_START = "2025-07-01"
SEASON_END = "2026-06-30"


def main():
    connection = PostgresConnection()
    embeddings = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))

    queue = VideoQueueManager(connection)
    queue.create_table()

    youtube_manager = YoutubeChannelVectorManager(
        connection=connection,
        embeddings=embeddings,
        chunk_size_seconds=30,
    )

    logger.info("Fetching existing ingested video IDs from vectorstore...")
    existing_ids = youtube_manager.get_existing_resource_ids()
    logger.info(f"  {len(existing_ids)} already ingested")

    logger.info(f"Fetching all videos from {CHANNEL} ({SEASON_START} to {SEASON_END})...")
    videos = youtube_manager.fetch_channel_videos(
        channel_handle=CHANNEL,
        published_after=SEASON_START,
        published_before=SEASON_END,
    )
    logger.info(f"  {len(videos)} videos found")

    inserted_pending = 0
    inserted_done = 0
    skipped = 0

    for video in videos:
        video_id = video["id"]
        publish_date = datetime.fromisoformat(
            video["publishedAt"].replace("Z", "+00:00")
        ).strftime("%Y-%m-%d")
        status = "done" if video_id in existing_ids else "pending"

        was_inserted = queue.upsert_video(
            video_id=video_id,
            title=video["title"],
            url=video["url"],
            publish_date=publish_date,
            status=status,
        )
        if was_inserted:
            if status == "done":
                inserted_done += 1
            else:
                inserted_pending += 1
        else:
            skipped += 1

    stats = queue.get_stats()
    logger.info("=" * 50)
    logger.info("Seed complete")
    logger.info(f"  Inserted pending : {inserted_pending}")
    logger.info(f"  Inserted done    : {inserted_done}")
    logger.info(f"  Skipped (exists) : {skipped}")
    logger.info(f"  Queue stats      : {stats}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
