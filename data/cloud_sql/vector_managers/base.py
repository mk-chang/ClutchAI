"""
Base class for vectorstore managers using PostgreSQL with pgvector.

This module provides the base functionality for managing resources in a PostgreSQL vectorstore,
including database initialization, URL management, document deletion, and YAML-based updates.
"""

from __future__ import annotations

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from abc import ABC, abstractmethod

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector

from data.cloud_sql.connection import PostgresConnection
from data.cloud_sql.schema import setup_pgvector_extension, get_default_table_name
from data.data_class import YouTubeVideo
from logger import get_logger

logger = get_logger(__name__)


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
        table_name: Optional[str] = None,
        vectordata_yaml: Optional[Path] = None,
    ):
        """
        Initialize the BaseVectorManager.
        
        Args:
            connection: PostgresConnection instance
            embeddings: OpenAIEmbeddings instance for document embedding
            table_name: Name of the vector table in PostgreSQL (defaults to env var CLOUDSQL_VECTOR_TABLE)
            vectordata_yaml: Path to YAML file with resource data (optional, can be passed to methods)
        """
        self.connection = connection
        self.embeddings = embeddings
        self.table_name = table_name or get_default_table_name()
        self.vectordata_yaml = vectordata_yaml
        
        # Ensure pgvector extension is set up (PGVector manages its own tables)
        setup_pgvector_extension(connection)
        
        # Initialize PGVector
        # PGVector will create its own tables (langchain_pg_collection, langchain_pg_embedding)
        # The collection_name parameter controls the collection name
        self.vectorstore = PGVector(
            embeddings=embeddings,
            connection=connection.get_engine(),
            collection_name=self.table_name,  # This becomes the collection name in langchain_pg_collection
            use_jsonb=True,
        )
    
    @staticmethod
    def get_env_var(
        env_var_name: str,
        param_value: Optional[str] = None,
        help_url: Optional[str] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Get environment variable value with helpful error messages.
        
        This method follows security best practices by:
        - Not storing API keys longer than necessary
        - Providing clear error messages when keys are missing
        - Allowing parameter override for flexibility
        
        Args:
            env_var_name: Name of the environment variable to check
            param_value: Optional parameter value (takes precedence over env var)
            help_url: Optional URL for getting the API key (for error messages)
            description: Optional description of what the key is for (for error messages)
            
        Returns:
            The API key value (from parameter or environment variable)
            
        Raises:
            ValueError: If the API key is not found in parameter or environment variable
        """
        # Check parameter first (allows explicit passing)
        if param_value:
            return param_value
        
        # Check environment variable
        env_value = os.environ.get(env_var_name)
        if env_value:
            return env_value
        
        # Build helpful error message
        error_parts = [
            f"API key is required for {description or 'this operation'}."
        ]
        
        if description:
            error_parts.append(f"Description: {description}")
        
        error_parts.append(f"Either provide the key as a parameter or set the {env_var_name} environment variable.")
        
        if help_url:
            error_parts.append(f"Get API key from: {help_url}")
        else:
            error_parts.append(f"See env.example for reference.")
        
        raise ValueError(" ".join(error_parts))
    
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
            logger.warning(f"Could not retrieve existing resource IDs from vectorstore: {e}")
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
            logger.warning(f"Error deleting documents for resource_id {resource_id}: {e}")
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
            
            # Auto-generate resource_id if missing (for articles, YouTube auto-extracts in dataclass)
            if not resource_id:
                if source_type == 'article':
                    # Generate hash-based ID from URL for articles
                    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                    resource_id = f"art{url_hash}"
                    logger.info(f"Auto-generated resource_id for article: {resource_id}")
                else:
                    logger.warning(f"Skipping {title} - resource_id is required but not provided")
                    results['failed'] += 1
                    continue
            
            # Check if resource already exists in vectorstore
            resource_exists = resource_id in existing_resource_ids
            was_updated = False
            
            # Handle force_update: delete existing documents first
            if force_update and resource_exists:
                logger.info(f"Force update requested for {title}. Deleting existing documents...")
                deleted_count = self.delete_documents_by_identifier(resource_id)
                if deleted_count > 0:
                    results['chunks_deleted'] += deleted_count
                    was_updated = True
                    logger.info(f"  ✓ Deleted {deleted_count} existing chunks")
                # Remove from existing_resource_ids so it will be re-added
                existing_resource_ids.discard(resource_id)
                resource_exists = False
            
            # Skip if resource exists and not forcing update
            if skip_existing and resource_exists and not force_update:
                logger.info(f"Skipping {title} (already in vectorstore, force_update=False)")
                results['skipped'] += 1
                continue
            
            try:
                action = "Updating" if was_updated else "Adding"
                logger.info(f"{action} {title} to vectorstore...")
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
                    logger.info(f"  ✓ Added {chunks_added} chunks")
                else:
                    results['failed'] += 1
                    logger.warning(f"  ✗ Failed to add (no content)")
            except Exception as e:
                results['failed'] += 1
                logger.error(f"  ✗ Failed to add {title}: {e}")
        
        return results
