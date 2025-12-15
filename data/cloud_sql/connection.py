"""
Cloud SQL PostgreSQL Connection Management using Google Cloud SQL Python Connector

Handles connection to Cloud SQL PostgreSQL instances using SQLAlchemy with Cloud SQL Python Connector.
Supports IAM authentication and simplified connection management.
"""

import os
from typing import Optional
from sqlalchemy import create_engine, Engine
from google.cloud.sql.connector import Connector


class PostgresConnection:
    """
    Wrapper for Cloud SQL PostgreSQL connections using SQLAlchemy.
    
    Uses Google Cloud SQL Python Connector for secure connections without password management.
    Works with Google Cloud SQL for PostgreSQL instances.
    """
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        region: Optional[str] = None,
        instance: Optional[str] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """
        Initialize PostgreSQL connection using Cloud SQL Python Connector.
        
        Args:
            project_id: GCP project ID (or from GOOGLE_CLOUD_PROJECT env var)
            region: Cloud SQL instance region (or from CLOUDSQL_REGION env var)
            instance: Cloud SQL instance name (or from CLOUDSQL_INSTANCE env var)
            database: Database name (or from CLOUDSQL_VECTOR_DATABASE env var)
            user: Database user (or from CLOUDSQL_USER env var)
            password: Database password (or from CLOUDSQL_PASSWORD env var, optional for IAM)
        """
        # Get connection parameters from args or environment
        self.project_id = project_id or os.environ.get('GOOGLE_CLOUD_PROJECT')
        self.region = region or os.environ.get('CLOUDSQL_REGION')
        self.instance = instance or os.environ.get('CLOUDSQL_INSTANCE')
        self.vectordb = database or os.environ.get('CLOUDSQL_VECTOR_DATABASE')
        self.user = user or os.environ.get('CLOUDSQL_USER')
        self.password = password or os.environ.get('CLOUDSQL_PASSWORD')
        
        if not all([self.project_id, self.region, self.instance, self.vectordb]):
            raise ValueError(
                "Cloud SQL connection parameters required. "
                "Provide project_id, region, instance, database or set environment variables: "
                "GOOGLE_CLOUD_PROJECT, CLOUDSQL_REGION, CLOUDSQL_INSTANCE, CLOUDSQL_VECTOR_DATABASE"
            )
        
        # Initialize Cloud SQL Connector
        self.connector = Connector()
        
        # Build connection string for SQLAlchemy
        # Format: postgresql+pg8000://user:password@/database?instance=project:region:instance
        connection_string = f"postgresql+pg8000://"
        if self.user:
            if self.password:
                connection_string += f"{self.user}:{self.password}@"
            else:
                connection_string += f"{self.user}@"
        connection_string += f"/{self.vectordb}"
        
        # Create SQLAlchemy engine with Cloud SQL Connector
        def getconn():
            return self.connector.connect(
                f"{self.project_id}:{self.region}:{self.instance}",
                "pg8000",
                user=self.user,
                password=self.password,
                db=self.vectordb,
            )
        
        self.engine = create_engine(
            "postgresql+pg8000://",
            creator=getconn,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
    
    def get_engine(self) -> Engine:
        """
        Get the SQLAlchemy Engine instance.
        
        Returns:
            SQLAlchemy Engine instance
        """
        return self.engine
    
    def get_connection_string(self) -> str:
        """
        Get a connection string for langchain-postgres.
        
        Note: langchain-postgres uses psycopg, so we need to provide
        a connection string that works with the Cloud SQL Connector.
        For direct use, you may need to use the engine directly.
        
        Returns:
            Connection string (for reference, but engine is preferred)
        """
        # langchain-postgres can use the engine directly via connection parameter
        # This is mainly for documentation
        return f"postgresql://{self.user or ''}:{self.password or ''}@/{self.vectordb}"
    
    def close(self):
        """Close the connector and engine."""
        if hasattr(self, 'connector') and self.connector:
            self.connector.close()
        if hasattr(self, 'engine') and self.engine:
            self.engine.dispose()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
