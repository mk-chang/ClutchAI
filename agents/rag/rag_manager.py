"""
RAG Manager for ClutchAI

This module provides RAG (Retrieval-Augmented Generation) capabilities using PostgreSQL with pgvector.
It focuses on reading from the vector database for retrieval operations only.

Architecture:
- RAGManager: Manages pgvector vectorstore for retrieval operations
- Provides LangChain-compatible retriever interface
"""

from __future__ import annotations

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

from data.cloud_sql.connection import PostgresConnection
from data.cloud_sql.schema import get_table_stats, get_default_table_name


class RAGManager:
    """
    Manager for RAG operations using PostgreSQL with pgvector.
    
    This class provides read-only access to the vector database for retrieval operations.
    All data ingestion should be handled by the data pipeline in data/cloud_sql/vector_managers.py.
    
    Features:
    - PostgreSQL/pgvector vectorstore integration
    - LangChain-compatible retriever interface
    - Vector similarity search
    - Statistics and metadata retrieval
    """
    
    @staticmethod
    def _load_config(project_root: Optional[Path] = None) -> Dict[str, Any]:
        """
        Load RAG configuration from rag_config.yaml.
        
        Args:
            project_root: Path to project root (defaults to detecting from file location)
            
        Returns:
            Dictionary with config values, or empty dict if file not found
        """
        if project_root is None:
            # Try to detect project root from this file's location
            # This file is at agents/rag/rag_manager.py
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent
        
        config_path = project_root / 'agents' / 'rag' / 'rag_config.yaml'
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def __init__(
        self,
        connection: Optional[PostgresConnection] = None,
        table_name: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        embedding_model: Optional[str] = None,
        k: Optional[int] = None,
        search_type: Optional[str] = None,
        project_root: Optional[Path] = None,
    ):
        """
        Initialize the RAG Manager.
        
        Args:
            connection: PostgresConnection instance (optional, will create if not provided)
            table_name: Name of the vector table in PostgreSQL (defaults to env var CLOUDSQL_VECTOR_TABLE)
            openai_api_key: OpenAI API key for embeddings (or from env)
            embedding_model: OpenAI embedding model to use (overrides config if provided)
            k: Number of documents to retrieve (overrides config if provided)
            search_type: Search type - "similarity" or "mmr" (overrides config if provided)
            project_root: Path to project root for loading config (optional, auto-detected)
        """
        # Load config from YAML file
        config = self._load_config(project_root)
        
        # Set API key
        self.openai_api_key = openai_api_key or os.environ.get('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY env var or pass openai_api_key parameter.")
        
        # Get embedding model from config or parameter (with fallback)
        embeddings_config = config.get('embeddings', {})
        model = embedding_model or embeddings_config.get('model', "text-embedding-ada-002")
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            api_key=self.openai_api_key,
            model=model
        )
        
        # Get or create connection
        if connection:
            self.connection = connection
        else:
            # Create connection from environment variables
            self.connection = PostgresConnection()
        
        self.table_name = table_name or get_default_table_name()
        
        # Get retrieval settings from config or parameters (with fallbacks)
        vectorstore_config = config.get('vectorstore', {})
        retrieval_config = config.get('retrieval', {})
        
        self.k = k if k is not None else vectorstore_config.get('k', 4)
        self.search_type = search_type if search_type is not None else retrieval_config.get('search_type', "similarity")
        
        # Store MMR config for later use if needed
        self.mmr_config = retrieval_config.get('mmr', {})
        
        # Initialize vectorstore (lazy loading)
        self._vectorstore: Optional[PGVector] = None
        self._retriever: Optional[BaseRetriever] = None
    
    def _get_vectorstore(self) -> PGVector:
        """
        Get or create the PGVector instance.
        
        Returns:
            PGVector instance
        """
        if self._vectorstore is None:
            # Use the existing SQLAlchemy Engine from PostgresConnection
            # This engine uses pg8000 driver which works with Cloud SQL Connector
            # and is compatible with langchain_postgres PGVector
            engine = self.connection.get_engine()
            
            # PGVector expects a SQLAlchemy Engine instance
            self._vectorstore = PGVector(
                embeddings=self.embeddings,
                connection=engine,
                collection_name=self.table_name,
                use_jsonb=True,
            )
        
        return self._vectorstore
    
    def get_retriever(
        self,
        k: Optional[int] = None,
        search_type: Optional[str] = None,
        search_kwargs: Optional[Dict[str, Any]] = None
    ) -> BaseRetriever:
        """
        Get a LangChain retriever for the vectorstore.
        
        Args:
            k: Number of documents to retrieve (uses instance default if not provided)
            search_type: Search type - "similarity" or "mmr" (uses instance default if not provided)
            search_kwargs: Additional search parameters (e.g., {"score_threshold": 0.7})
            
        Returns:
            LangChain BaseRetriever instance
        """
        vectorstore = self._get_vectorstore()
        
        retriever_k = k if k is not None else self.k
        retriever_search_type = search_type if search_type is not None else self.search_type
        
        if retriever_search_type == "mmr":
            # Use MMR config from config file if available
            mmr_kwargs = {}
            if hasattr(self, 'mmr_config') and self.mmr_config:
                if 'lambda_param' in self.mmr_config:
                    mmr_kwargs['lambda_param'] = self.mmr_config['lambda_param']
                if 'fetch_k' in self.mmr_config:
                    mmr_kwargs['fetch_k'] = self.mmr_config['fetch_k']
            
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": retriever_k,
                    **mmr_kwargs,
                    **(search_kwargs or {})
                }
            )
        else:
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": retriever_k,
                    **(search_kwargs or {})
                }
            )
        
        return retriever
    
    def retrieve(self, query: str, k: Optional[int] = None) -> list[Document]:
        """
        Retrieve documents similar to the query.
        
        Args:
            query: Query string
            k: Number of documents to retrieve (uses instance default if not provided)
            
        Returns:
            List of Document objects
        """
        retriever = self.get_retriever(k=k)
        return retriever.invoke(query)
    
    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> list[Document]:
        """
        Perform similarity search on the vectorstore.
        
        Args:
            query: Query string
            k: Number of documents to retrieve (uses instance default if not provided)
            filter: Optional metadata filter dictionary
            
        Returns:
            List of Document objects
        """
        vectorstore = self._get_vectorstore()
        search_k = k if k is not None else self.k
        
        if filter:
            return vectorstore.similarity_search(query, k=search_k, filter=filter)
        else:
            return vectorstore.similarity_search(query, k=search_k)
    
    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> list[tuple[Document, float]]:
        """
        Perform similarity search with scores.
        
        Args:
            query: Query string
            k: Number of documents to retrieve (uses instance default if not provided)
            filter: Optional metadata filter dictionary
            
        Returns:
            List of tuples (Document, score)
        """
        vectorstore = self._get_vectorstore()
        search_k = k if k is not None else self.k
        
        if filter:
            return vectorstore.similarity_search_with_score(query, k=search_k, filter=filter)
        else:
            return vectorstore.similarity_search_with_score(query, k=search_k)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vectorstore.
        
        Returns:
            Dictionary with vectorstore statistics
        """
        if not hasattr(self, 'connection') or self.connection is None:
            # Create temporary connection for stats
            connection = PostgresConnection()
        else:
            connection = self.connection
        
        try:
            stats = get_table_stats(connection, table_name=self.table_name)
            
            # Add additional stats
            stats['table_name'] = self.table_name
            stats['embedding_model'] = self.embeddings.model
            
            return stats
        except Exception as e:
            return {'error': str(e)}
    
    @property
    def vectorstore(self) -> PGVector:
        """
        Get the vectorstore instance.
        
        Returns:
            PGVector instance
        """
        return self._get_vectorstore()
    
    @property
    def retriever(self) -> BaseRetriever:
        """
        Get the default retriever instance.
        
        Returns:
            LangChain BaseRetriever instance
        """
        if self._retriever is None:
            self._retriever = self.get_retriever()
        return self._retriever
