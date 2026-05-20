import os
from sqlalchemy import create_engine, Engine


class PostgresConnection:
    """
    PostgreSQL connection using a standard DATABASE_URL.
    Provides a SQLAlchemy engine compatible with langchain-postgres PGVector.
    """

    def __init__(self, database_url: str = None):
        url = database_url or os.environ.get('DATABASE_URL')
        if not url:
            raise ValueError(
                "DATABASE_URL environment variable is required. "
                "Railway injects this automatically when a PostgreSQL plugin is attached."
            )
        # Railway provides postgresql:// URLs; SQLAlchemy needs the driver specified.
        if url.startswith('postgresql://'):
            url = url.replace('postgresql://', 'postgresql+psycopg2://', 1)
        self._engine = create_engine(url, pool_pre_ping=True, pool_recycle=3600)

    def get_engine(self) -> Engine:
        return self._engine

    def close(self):
        if hasattr(self, '_engine') and self._engine:
            self._engine.dispose()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
