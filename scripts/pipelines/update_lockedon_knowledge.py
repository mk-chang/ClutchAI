"""
Daily cron: check for new LockedOn videos + process pending queue entries.

Run:
    python scripts/pipelines/update_lockedon_knowledge.py
"""

import os
import sys
import time
import yaml
from datetime import datetime, timedelta
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from data.postgres.connection import PostgresConnection
from data.postgres.video_queue import AllTranscriptSourcesExhausted, VideoQueueManager
from data.postgres.vector_managers import YoutubeChannelVectorManager
from langchain_openai import OpenAIEmbeddings
from logger import get_logger, setup_logging

setup_logging(debug=False)
logger = get_logger(__name__)

vector_config_path = project_root / "config" / "vector_config.yaml"
vector_config = {}
if vector_config_path.exists():
    with open(vector_config_path) as f:
        vector_config = yaml.safe_load(f) or {}

rag_config_path = project_root / "config" / "rag_config.yaml"
rag_config = {}
if rag_config_path.exists():
    with open(rag_config_path) as f:
        rag_config = yaml.safe_load(f) or {}

CHANNEL = "@LockedOnFantasyBasketball"
DEV_MODE = os.environ.get("DEV_MODE", "false").lower() == "true"


def main():
    connection = PostgresConnection()
    embeddings = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))

    queue = VideoQueueManager(connection)
    queue.create_table()  # no-op if already exists

    youtube_manager = YoutubeChannelVectorManager(
        connection=connection,
        embeddings=embeddings,
        chunk_size_seconds=rag_config.get("youtube", {}).get("chunk_size_seconds", 30),
    )

    youtube_config = vector_config.get("youtube_channel", {})
    max_videos_added = 3 if DEV_MODE else youtube_config.get("max_videos_added", 15)
    delay_between_videos = youtube_config.get("delay_between_videos", 15.0)

    # Step 1: Enqueue any new videos published yesterday
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    logger.info(f"Checking for new videos published on {yesterday}...")
    try:
        new_videos = youtube_manager.fetch_channel_videos(
            channel_handle=CHANNEL,
            published_after=yesterday,
            published_before=yesterday,
        )
        for video in new_videos:
            publish_date = datetime.fromisoformat(
                video["publishedAt"].replace("Z", "+00:00")
            ).strftime("%Y-%m-%d")
            added = queue.upsert_video(
                video_id=video["id"],
                title=video["title"],
                url=video["url"],
                publish_date=publish_date,
            )
            if added:
                logger.info(f"  + Enqueued: {video['title']}")
    except Exception as e:
        logger.warning(f"  Could not check for new videos: {e}")

    # Step 2: Process pending videos
    pending = queue.get_pending(limit=max_videos_added)
    logger.info(f"\nProcessing {len(pending)} pending videos (max={max_videos_added})...")

    if not pending:
        logger.info("Queue is empty — nothing to do.")
        logger.info(f"Queue stats: {queue.get_stats()}")
        return

    videos_added = 0
    for i, video in enumerate(pending, 1):
        video_id = str(video["video_id"])
        title = video["title"] or video_id
        url = str(video["url"])
        publish_date = str(video["publish_date"]) if video["publish_date"] else None

        logger.info(f"\n[{i}/{len(pending)}] {title}")

        try:
            chunks_added = youtube_manager.add_resource_to_vectorstore(
                url=url,
                source_type="youtube",
                title=title,
                publish_date=publish_date,
                resource_id=video_id,
            )
            if chunks_added > 0:
                queue.mark_done(video_id)
                videos_added += 1
                logger.info(f"  ✓ Added {chunks_added} chunks")
            else:
                queue.mark_failed(video_id)
                logger.warning("  ✗ No transcript found")
        except AllTranscriptSourcesExhausted as e:
            queue.mark_failed(video_id)
            logger.error(f"  ✗ All sources exhausted: {e}")
            logger.error("  Stopping run — will retry tomorrow.")
            break
        except Exception as e:
            queue.mark_failed(video_id)
            logger.error(f"  ✗ Failed: {e}")

        if i < len(pending) and delay_between_videos > 0:
            time.sleep(delay_between_videos)

    logger.info("\n" + "=" * 50)
    logger.info(f"Videos added this run : {videos_added}")
    logger.info(f"Queue stats           : {queue.get_stats()}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
