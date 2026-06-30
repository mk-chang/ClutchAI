import json
from typing import Optional

from sqlalchemy import text

from data.postgres.connection import PostgresConnection
from logger import get_logger

logger = get_logger(__name__)


class WaiverWireStore:
    TABLE = "waiver_wire_cache"

    def __init__(self, connection: PostgresConnection):
        self.connection = connection

    def create_table(self) -> None:
        engine = self.connection.get_engine()
        with engine.connect() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {self.TABLE} (
                    league_key  VARCHAR(50)              PRIMARY KEY,
                    players     JSONB                    NOT NULL,
                    last_tx_id  INTEGER,
                    fetched_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """))
            conn.commit()
        logger.debug(f"Table '{self.TABLE}' ready")

    def get(self, league_key: str) -> Optional[dict]:
        """Return {'players': [...], 'last_tx_id': int|None} or None if no row."""
        engine = self.connection.get_engine()
        with engine.connect() as conn:
            row = conn.execute(
                text(f"SELECT players, last_tx_id FROM {self.TABLE} WHERE league_key = :k"),
                {"k": league_key},
            ).fetchone()
        if row is None:
            return None
        return {"players": row.players, "last_tx_id": row.last_tx_id}

    def put(self, league_key: str, players: list, last_tx_id: Optional[int]) -> None:
        engine = self.connection.get_engine()
        with engine.connect() as conn:
            conn.execute(
                text(f"""
                    INSERT INTO {self.TABLE} (league_key, players, last_tx_id, fetched_at)
                    VALUES (:k, :p::jsonb, :tx, NOW())
                    ON CONFLICT (league_key) DO UPDATE
                    SET players    = EXCLUDED.players,
                        last_tx_id = EXCLUDED.last_tx_id,
                        fetched_at = EXCLUDED.fetched_at
                """),
                {"k": league_key, "p": json.dumps(players), "tx": last_tx_id},
            )
            conn.commit()

    def delete(self, league_key: str) -> None:
        engine = self.connection.get_engine()
        with engine.connect() as conn:
            conn.execute(
                text(f"DELETE FROM {self.TABLE} WHERE league_key = :k"),
                {"k": league_key},
            )
            conn.commit()
