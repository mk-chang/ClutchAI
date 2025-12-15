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
from pathlib import Path
from typing import Optional, Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

from data.cloud_sql.connection import PostgresConnection
from data.cloud_sql.schema import get_table_stats


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
    
    def __init__(
        self,
        connection: Optional[PostgresConnection] = None,
        table_name: str = "embeddings",
        openai_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-ada-002",
        k: int = 4,
        search_type: str = "similarity",
    ):
        """
        Initialize the RAG Manager.
        
        Args:
            connection: PostgresConnection instance (optional, will create if not provided)
            table_name: Name of the vector table in PostgreSQL (default: "embeddings")
            openai_api_key: OpenAI API key for embeddings (or from env)
            embedding_model: OpenAI embedding model to use (default: "text-embedding-ada-002")
            k: Number of documents to retrieve (default: 4)
            search_type: Search type - "similarity" or "mmr" (default: "similarity")
        """
        # Set API key
        self.openai_api_key = openai_api_key or os.environ.get('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY env var or pass openai_api_key parameter.")
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            api_key=self.openai_api_key,
            model=embedding_model
        )
        
        # Get or create connection
        if connection:
            self.connection = connection
        else:
            # Create connection from environment variables
            self.connection = PostgresConnection()
        
        self.table_name = table_name
        self.k = k
        self.search_type = search_type
        
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
            # Get connection parameters
            project_id = self.connection.project_id
            region = self.connection.region
            instance = self.connection.instance
            database = self.connection.vectordb
            user = self.connection.user
            password = self.connection.password
            
            # PGVector from langchain-postgres uses psycopg2
            # We need to create a connection using Cloud SQL Connector with psycopg2
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
            
            # PGVector can accept a connection callable
            # We'll use the connection parameter with our factory
            self._vectorstore = PGVector(
                embeddings=self.embeddings,
                connection=get_connection,
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
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": retriever_k,
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
