import pytest
from data.cloud_sql.connection import PostgresConnection


def test_raises_without_database_url(monkeypatch):
    monkeypatch.delenv('DATABASE_URL', raising=False)
    with pytest.raises(ValueError, match="DATABASE_URL"):
        PostgresConnection()


def test_creates_engine_from_database_url(monkeypatch):
    monkeypatch.setenv('DATABASE_URL', 'postgresql://user:pass@localhost:5432/testdb')
    conn = PostgresConnection()
    assert conn.get_engine() is not None
    conn.close()


def test_accepts_explicit_database_url(monkeypatch):
    monkeypatch.delenv('DATABASE_URL', raising=False)
    conn = PostgresConnection(database_url='postgresql://user:pass@localhost:5432/testdb')
    assert conn.get_engine() is not None
    conn.close()


def test_rewrites_postgresql_url_to_psycopg2(monkeypatch):
    monkeypatch.setenv('DATABASE_URL', 'postgresql://user:pass@localhost:5432/testdb')
    conn = PostgresConnection()
    assert 'psycopg2' in str(conn.get_engine().url)
    conn.close()


def test_context_manager(monkeypatch):
    monkeypatch.setenv('DATABASE_URL', 'postgresql://user:pass@localhost:5432/testdb')
    with PostgresConnection() as conn:
        assert conn.get_engine() is not None
