"""
Base class for vectorstore managers using PostgreSQL with pgvector.

This module provides the base functionality for managing resources in a PostgreSQL vectorstore,
including database initialization, URL management, document deletion, and YAML-based updates.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from abc import ABC, abstractmethod

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector

from data.cloud_sql.connection import PostgresConnection
from data.cloud_sql.schema import setup_schema, create_vector_table
from data.data_class import YouTubeVideo


class BaseVectorManager(ABC):
    """
    Base class for vectorstore managers using PostgreSQL with pgvector.
    
    This class provides common functionality for managing resources in a PostgreSQL vectorstore,
    including database initialization, URL management, document deletion, and YAML-based updates.
    """
    
    def __init__(
        self,
        connection: PostgresConnection,
        embeddings: OpenAIEmbeddings,
        table_name: str = "embeddings",
        vectordata_yaml: Optional[Path] = None,
    ):
        """
        Initialize the BaseVectorManager.
        
        Args:
            connection: PostgresConnection instance
            embeddings: OpenAIEmbeddings instance for document embedding
            table_name: Name of the vector table in PostgreSQL
            vectordata_yaml: Path to YAML file with resource data (optional, can be passed to methods)
        """
        self.connection = connection
        self.embeddings = embeddings
        self.table_name = table_name
        self.vectordata_yaml = vectordata_yaml
        
        # Ensure schema is set up
        setup_schema(connection, vector_size=1536)  # OpenAI embedding dimension
        create_vector_table(connection, table_name=table_name, vector_dimension=1536)
        
        # Initialize PGVector
        # Get connection parameters
        project_id = connection.project_id
        region = connection.region
        instance = connection.instance
        database = connection.vectordb
        user = connection.user
        password = connection.password
        
        from google.cloud.sql.connector import Connector
        
        connector = Connector()
        
        # Create a connection factory for PGVector
        def get_connection():
            """Create a psycopg2 connection using Cloud SQL Connector."""
            return connector.connect(
                f"{project_id}:{region}:{instance}",
                "psycopg2",
                user=user,
                password=password,
                db=database,
            )
        
        self.vectorstore = PGVector(
            embeddings=embeddings,
            connection=get_connection,
            collection_name=table_name,
            use_jsonb=True,
        )
    
    def normalize_url(self, url: str) -> str:
        """
        Normalize URL for comparison.
        
        Base implementation removes timestamps. Subclasses can override for additional normalization.
        
        Args:
            url: URL to normalize
            
        Returns:
            Normalized URL
        """
        # Remove timestamp parameters: &t=, #t=, &t (without value), etc.
        normalized = url.split('&t=')[0].split('#t=')[0]
        # Handle &t without = (e.g., &t at end of URL)
        if '&t' in normalized and '&t=' not in normalized:
            normalized = normalized.split('&t')[0]
        return normalized
    
    @classmethod
    def _ensure_string_date(cls, date_value) -> Optional[str]:
        """
        Convert date value to string format for database storage.
        
        Args:
            date_value: Date value (can be str, date, datetime, or None)
            
        Returns:
            String representation of date in YYYY-MM-DD format, or None
        """
        if date_value is None:
            return None
        
        # If already a string, return as-is
        if isinstance(date_value, str):
            return date_value
        
        # If it's a date or datetime object, convert to string
        if hasattr(date_value, 'strftime'):
            return date_value.strftime('%Y-%m-%d')
        
        # Fallback: convert to string
        return str(date_value)
    
    def get_existing_resource_ids(self) -> Set[str]:
        """
        Get resource IDs of resources already in the vectorstore.
        
        Returns:
            Set of resource IDs
        """
        try:
            # Use vectorstore's connection to query
            # PGVector stores metadata in JSONB, we need to query for resource_id
            # We'll query all documents and extract resource_id from metadata
            # Note: This approach works but may be inefficient for large datasets
            
            # Alternative: Query all documents and check metadata
            # This is inefficient but works with the current API limitations
            all_docs = self.vectorstore.similarity_search("", k=10000)  # Get all docs (limit may need adjustment)
            resource_ids = set()
            for doc in all_docs:
                resource_id = doc.metadata.get('resource_id')
                if resource_id:
                    resource_ids.add(resource_id)
            
            return resource_ids
        except Exception as e:
            print(f"Warning: Could not retrieve existing resource IDs from vectorstore: {e}")
            return set()
    
    def delete_documents_by_identifier(self, resource_id: str) -> int:
        """
        Delete all documents from vectorstore that match the given resource ID.
        
        Args:
            resource_id: Unique resource identifier (required)
            
        Returns:
            Number of documents deleted
        """
        if not resource_id:
            return 0
        
        try:
            # PGVector doesn't have a direct delete by metadata method
            # We need to find documents by resource_id and delete them
            # Query for documents with this resource_id
            docs_to_delete = self.vectorstore.similarity_search(
                "",  # Empty query to get all
                k=10000,  # Large limit
                filter={"resource_id": resource_id}
            )
            
            deleted_count = 0
            for doc in docs_to_delete:
                # Delete by document ID if available
                if hasattr(doc, 'id') and doc.id:
                    self.vectorstore.delete([doc.id])
                    deleted_count += 1
                elif 'id' in doc.metadata:
                    self.vectorstore.delete([doc.metadata['id']])
                    deleted_count += 1
            
            return deleted_count
        except Exception as e:
            print(f"Warning: Error deleting documents for resource_id {resource_id}: {e}")
            # Fallback: try to use engine's connection for direct SQL delete
            try:
                # This is a workaround - PGVector may not expose direct SQL delete
                # We may need to access the underlying connection for more efficient deletion
                return 0
            except:
                return 0
    
    def _add_documents_to_vectorstore(self, docs: List[Document]) -> None:
        """
        Add documents to the PostgreSQL vectorstore using PGVector.
        
        Args:
            docs: List of Document objects to add
        """
        if not docs:
            return
        
        # PGVector handles embeddings automatically
        # Just add documents - embeddings will be generated automatically
        self.vectorstore.add_documents(docs)
    
    @abstractmethod
    def load_resources_from_yaml(self, vectordata_yaml: Optional[Path] = None) -> List[Tuple[str, YouTubeVideo]]:
        """
        Load resources from YAML file.
        
        Args:
            vectordata_yaml: Path to YAML file (uses self.vectordata_yaml if not provided)
            
        Returns:
            List of tuples (source_type, YouTubeVideo) where source_type is the YAML top-level key
        """
        pass
    
    @abstractmethod
    def load_resource_content(
        self,
        url: str,
        source_type: str,
        title: Optional[str] = None,
        upload_date: Optional[str] = None,
        publish_date: Optional[str] = None,
        resource_id: Optional[str] = None,
        **kwargs
    ) -> List[Document]:
        """
        Load and process resource content from a URL.
        
        Args:
            url: Resource URL
            source_type: Source type identifier from YAML top-level key
            title: Resource title (optional, for metadata)
            upload_date: Upload date in YYYY-MM-DD format (optional, for metadata)
            publish_date: Publish date in YYYY-MM-DD format (optional, for metadata)
            resource_id: Unique resource identifier (optional, for matching even if URL changes)
            **kwargs: Additional arguments specific to the resource type
            
        Returns:
            List of Document objects with resource chunks and enhanced metadata
        """
        pass
    
    @abstractmethod
    def add_resource_to_vectorstore(
        self,
        url: str,
        source_type: str,
        title: Optional[str] = None,
        upload_date: Optional[str] = None,
        publish_date: Optional[str] = None,
        resource_id: Optional[str] = None,
        **kwargs
    ) -> int:
        """
        Add a single resource to the vectorstore.
        
        Args:
            url: Resource URL
            source_type: Source type identifier from YAML top-level key
            title: Resource title (optional, for metadata)
            upload_date: Upload date in YYYY-MM-DD format (optional, for metadata)
            publish_date: Publish date in YYYY-MM-DD format (optional, for metadata)
            resource_id: Unique resource identifier (optional, for matching even if URL changes)
            **kwargs: Additional arguments specific to the resource type
            
        Returns:
            Number of document chunks added
        """
        pass
    
    def update_vectorstore_from_yaml(
        self,
        vectordata_yaml: Optional[Path] = None,
        skip_existing: bool = True,
        **kwargs
    ) -> Dict[str, int]:
        """
        Update vectorstore with all resources from YAML file.
        
        Args:
            vectordata_yaml: Path to YAML file (uses self.vectordata_yaml if not provided)
            skip_existing: Skip resources that are already in the vectorstore (unless force_update is True)
            **kwargs: Additional arguments specific to the resource type (e.g., chunk_size_seconds for YouTube)
            
        Returns:
            Dictionary with 'added', 'skipped', 'failed', 'updated', 'chunks_added', 'chunks_deleted' counts
        """
        # Load resources from YAML
        resources_with_source = self.load_resources_from_yaml(vectordata_yaml)
        
        # Get existing resource IDs if skipping
        existing_resource_ids = set()
        if skip_existing:
            existing_resource_ids = self.get_existing_resource_ids()
        
        results = {
            'added': 0,
            'skipped': 0,
            'failed': 0,
            'updated': 0,
            'chunks_added': 0,
            'chunks_deleted': 0
        }
        
        for source_type, resource in resources_with_source:
            url = resource.url
            resource_id = getattr(resource, 'id', None)
            title = resource.title
            upload_date = resource.upload_date
            publish_date = getattr(resource, 'publish_date', None) or upload_date
            force_update = resource.force_update
            
            # Require resource_id for identification
            if not resource_id:
                print(f"Warning: Skipping {title} - resource_id is required but not provided")
                results['failed'] += 1
                continue
            
            # Check if resource already exists in vectorstore
            resource_exists = resource_id in existing_resource_ids
            was_updated = False
            
            # Handle force_update: delete existing documents first
            if force_update and resource_exists:
                print(f"Force update requested for {title}. Deleting existing documents...")
                deleted_count = self.delete_documents_by_identifier(resource_id)
                if deleted_count > 0:
                    results['chunks_deleted'] += deleted_count
                    was_updated = True
                    print(f"  ✓ Deleted {deleted_count} existing chunks")
                # Remove from existing_resource_ids so it will be re-added
                existing_resource_ids.discard(resource_id)
                resource_exists = False
            
            # Skip if resource exists and not forcing update
            if skip_existing and resource_exists and not force_update:
                print(f"Skipping {title} (already in vectorstore, force_update=False)")
                results['skipped'] += 1
                continue
            
            try:
                action = "Updating" if was_updated else "Adding"
                print(f"{action} {title} to vectorstore...")
                chunks_added = self.add_resource_to_vectorstore(
                    url,
                    source_type=source_type,
                    title=title,
                    upload_date=upload_date,
                    publish_date=publish_date,
                    resource_id=resource_id,
                    **kwargs
                )
                
                if chunks_added > 0:
                    if was_updated:
                        # This was a force update - count as updated
                        results['updated'] += 1
                    else:
                        results['added'] += 1
                    results['chunks_added'] += chunks_added
                    # Update existing_resource_ids to avoid duplicates
                    existing_resource_ids.add(resource_id)
                    print(f"  ✓ Added {chunks_added} chunks")
                else:
                    results['failed'] += 1
                    print(f"  ✗ Failed to add (no content)")
            except Exception as e:
                results['failed'] += 1
                print(f"  ✗ Failed to add {title}: {e}")
        
        return results
