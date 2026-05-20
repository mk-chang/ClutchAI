import os
from sqlalchemy import create_engine, Engine


class PostgresConnection:

    def __init__(self, database_url: str = None):
        url = database_url or os.environ.get('DATABASE_URL')
        if not url:
            raise ValueError(
                "DATABASE_URL environment variable is required. "
                "Railway injects this automatically when a PostgreSQL plugin is attached."
            )
        if url.startswith('postgresql://'):
            url = url.replace('postgresql://', 'postgresql+psycopg2://', 1)
        self._engine = create_engine(url, pool_pre_ping=True, pool_recycle=3600)

    def get_engine(self) -> Engine:
        return self._engine

    def close(self):
        self._engine.dispose()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
