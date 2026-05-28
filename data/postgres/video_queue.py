from __future__ import annotations

from typing import Dict, List, Optional

from sqlalchemy import text

from data.postgres.connection import PostgresConnection
from logger import get_logger

logger = get_logger(__name__)


class AllTranscriptSourcesExhausted(Exception):
    """Both YoutubeLoader and Supadata failed — the run should stop."""


class VideoQueueManager:
    TABLE = "lockedon_video_queue"

    def __init__(self, connection: PostgresConnection):
        self.connection = connection

    def create_table(self) -> None:
        engine = self.connection.get_engine()
        with engine.connect() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {self.TABLE} (
                    video_id     VARCHAR(20) PRIMARY KEY,
                    title        TEXT,
                    url          TEXT NOT NULL,
                    publish_date DATE,
                    status       VARCHAR(10) DEFAULT 'pending'
                                     CHECK (status IN ('pending', 'done', 'failed')),
                    attempts     INTEGER DEFAULT 0,
                    last_attempted_at TIMESTAMP WITH TIME ZONE,
                    added_at     TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """))
            conn.commit()
        logger.info(f"Table '{self.TABLE}' ready")

    def upsert_video(
        self,
        video_id: str,
        title: str,
        url: str,
        publish_date: Optional[str] = None,
        status: str = "pending",
    ) -> bool:
        """Insert video if not already in queue. Returns True if inserted."""
        engine = self.connection.get_engine()
        with engine.connect() as conn:
            result = conn.execute(
                text(f"""
                    INSERT INTO {self.TABLE} (video_id, title, url, publish_date, status)
                    VALUES (:video_id, :title, :url, :publish_date, :status)
                    ON CONFLICT (video_id) DO NOTHING
                """),
                {
                    "video_id": video_id,
                    "title": title,
                    "url": url,
                    "publish_date": publish_date,
                    "status": status,
                },
            )
            conn.commit()
            return result.rowcount > 0

    def get_pending(self, limit: int = 15) -> List[Dict]:
        """Return up to `limit` pending videos, newest-first."""
        engine = self.connection.get_engine()
        with engine.connect() as conn:
            result = conn.execute(
                text(f"""
                    SELECT video_id, title, url, publish_date
                    FROM {self.TABLE}
                    WHERE status = 'pending'
                    ORDER BY publish_date DESC NULLS LAST
                    LIMIT :limit
                """),
                {"limit": limit},
            )
            return [dict(row._mapping) for row in result.fetchall()]

    def mark_done(self, video_id: str) -> None:
        engine = self.connection.get_engine()
        with engine.connect() as conn:
            conn.execute(
                text(f"""
                    UPDATE {self.TABLE}
                    SET status = 'done', last_attempted_at = NOW()
                    WHERE video_id = :video_id
                """),
                {"video_id": video_id},
            )
            conn.commit()

    def mark_failed(self, video_id: str) -> None:
        engine = self.connection.get_engine()
        with engine.connect() as conn:
            conn.execute(
                text(f"""
                    UPDATE {self.TABLE}
                    SET status = 'failed',
                        attempts = attempts + 1,
                        last_attempted_at = NOW()
                    WHERE video_id = :video_id
                """),
                {"video_id": video_id},
            )
            conn.commit()

    def get_stats(self) -> Dict[str, int]:
        engine = self.connection.get_engine()
        with engine.connect() as conn:
            result = conn.execute(
                text(f"SELECT status, COUNT(*) FROM {self.TABLE} GROUP BY status")
            )
            return {row[0]: row[1] for row in result.fetchall()}
