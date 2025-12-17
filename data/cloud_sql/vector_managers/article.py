"""
Article vector manager for ingesting articles into a PostgreSQL vectorstore.

This module provides managers for article-specific operations including:
- Loading articles from YAML configuration
- Scraping article content from URLs using Firecrawl
- Chunking article content
- Adding articles to vectorstore
"""

from __future__ import annotations

import os
import yaml
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from urllib.parse import urlparse

try:
    from firecrawl import Firecrawl
except ImportError:
    Firecrawl = None

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from data.cloud_sql.connection import PostgresConnection
from data.cloud_sql.vector_managers.base import BaseVectorManager
from logger import get_logger

logger = get_logger(__name__)
from data.data_class import YouTubeVideo


class ArticleVectorManager(BaseVectorManager):
    """
    Manager for ingesting articles into a PostgreSQL vectorstore.
    
    This class handles all article-specific operations including:
    - Loading articles from YAML configuration
    - Scraping article content from URLs
    - Chunking article content
    - Adding articles to vectorstore
    - Deleting articles from vectorstore
    - Handling force updates
    """
    
    def __init__(
        self,
        connection: PostgresConnection,
        embeddings: OpenAIEmbeddings,
        table_name: Optional[str] = None,
        vectordata_yaml: Optional[Path] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        firecrawl_api_key: Optional[str] = None,
    ):
        """
        Initialize the ArticleVectorManager.
        
        Args:
            connection: PostgresConnection instance
            embeddings: OpenAIEmbeddings instance for document embedding
            table_name: Name of the vector table in PostgreSQL
            vectordata_yaml: Path to YAML file with article data (optional, can be passed to methods)
            chunk_size: Size of text chunks in characters (default: 1000)
            chunk_overlap: Overlap between chunks in characters (default: 200)
            firecrawl_api_key: Firecrawl API key (optional, will use FIRECRAWL_API_KEY env var if not provided)
        """
        super().__init__(connection, embeddings, table_name, vectordata_yaml)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        # Initialize Firecrawl client
        if Firecrawl is None:
            raise ImportError(
                "firecrawl-py is not installed. Please install it with: pip install firecrawl-py"
            )
        
        # Get Firecrawl API key using base class utility
        # Note: We store this as instance variable because it's needed to initialize the Firecrawl client
        self.firecrawl_api_key = self.get_env_var(
            env_var_name='FIRECRAWL_API_KEY',
            param_value=firecrawl_api_key,
            help_url='https://firecrawl.dev/app/api-keys',
            description='Firecrawl API (for web scraping)'
        )
        
        self.firecrawl_client = Firecrawl(api_key=self.firecrawl_api_key)
    
    def normalize_url(self, url: str) -> str:
        """
        Normalize URL for comparison.
        
        Overrides base implementation to also strip trailing slashes for articles.
        
        Args:
            url: URL to normalize
            
        Returns:
            Normalized URL
        """
        normalized = super().normalize_url(url)
        return normalized.rstrip('/')
    
    def load_resources_from_yaml(self, vectordata_yaml: Optional[Path] = None) -> List[Tuple[str, YouTubeVideo]]:
        """
        Load articles from YAML file.
        
        Args:
            vectordata_yaml: Path to YAML file (uses self.vectordata_yaml if not provided)
            
        Returns:
            List of tuples (source_type, YouTubeVideo) where source_type is the YAML top-level key
            (e.g., 'article'). Note: Uses YouTubeVideo dataclass as it has the same structure.
        """
        yaml_path = vectordata_yaml or self.vectordata_yaml
        if not yaml_path or not yaml_path.exists():
            logger.warning(f"YAML file not found at {yaml_path}")
            return []
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                
                if not data:
                    logger.warning("YAML file is empty.")
                    return []
                
                articles = []
                # Iterate through all top-level keys, but only process 'article' key
                for source_type, items in data.items():
                    if source_type != 'article' or not isinstance(items, list):
                        continue
                    
                    for item_data in items:
                        if not isinstance(item_data, dict):
                            logger.warning(f"Skipping invalid entry in {source_type}: {item_data}")
                            continue
                        
                        # Ensure required fields
                        if 'title' not in item_data or 'url' not in item_data:
                            logger.warning(f"Skipping {source_type} entry without title or url: {item_data}")
                            continue
                        
                        article = YouTubeVideo.from_dict(item_data)  # Reuse same dataclass
                        articles.append((source_type, article))
                
                return articles
        except (yaml.YAMLError, KeyError, TypeError) as e:
            logger.warning(f"Error loading YAML file: {e}. Starting with empty list.")
            return []
    
    def scrape_article_content(self, url: str, max_content_length: Optional[int] = None) -> Dict[str, str]:
        """
        Scrape article content from a URL using Firecrawl.
        
        Args:
            url: Article URL to scrape
            max_content_length: Maximum length of content to return (optional)
            
        Returns:
            Dictionary with 'url', 'title', 'content', 'source', and 'publish_date' keys
        """
        try:
            # Parse and normalize URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme:
                url = f"https://{url}"
                parsed_url = urlparse(url)
            elif not parsed_url.netloc:
                raise ValueError(f"Invalid URL format: {url}")
            
            domain = parsed_url.netloc
            
            # Scrape the URL using Firecrawl
            # Use markdown format for clean text extraction
            result = self.firecrawl_client.scrape(url, formats=['markdown'])
            
            # Extract data from Firecrawl response
            # Handle both object attributes and dictionary access
            if hasattr(result, 'markdown'):
                markdown_content = result.markdown
                title = getattr(result, 'title', None) or 'Untitled'
                # Try to get publish date from various possible fields
                publish_date = (
                    getattr(result, 'publishedDate', None) or
                    getattr(result, 'datePublished', None) or
                    getattr(result, 'date', None) or
                    (hasattr(result, 'metadata') and getattr(result.metadata, 'publishedDate', None)) or
                    (hasattr(result, 'metadata') and getattr(result.metadata, 'datePublished', None))
                )
            elif isinstance(result, dict):
                markdown_content = result.get('markdown', '')
                title = result.get('title', 'Untitled')
                metadata = result.get('metadata', {})
                publish_date = (
                    result.get('publishedDate') or
                    result.get('datePublished') or
                    result.get('date') or
                    (metadata.get('publishedDate') if isinstance(metadata, dict) else None) or
                    (metadata.get('datePublished') if isinstance(metadata, dict) else None)
                )
            else:
                # Fallback: try to get as dict
                markdown_content = getattr(result, 'markdown', '') or str(result)
                title = getattr(result, 'title', 'Untitled')
                publish_date = None
            
            # Convert markdown to plain text if needed (remove markdown formatting)
            # For now, we'll keep markdown as it's cleaner than HTML
            content = markdown_content.strip()
            
            # Validate content
            if not content or len(content) < 50:
                raise ValueError(f"Minimal or no content extracted from {url}")
            
            # Truncate if max_content_length is specified
            if max_content_length and len(content) > max_content_length:
                content = content[:max_content_length] + "..."
            
            # Parse publish date if available
            parsed_publish_date = None
            if publish_date:
                try:
                    # Handle ISO format with timezone
                    date_str = str(publish_date).strip()
                    if 'T' in date_str:
                        date_str = date_str.split('T')[0]
                    # Try parsing common formats
                    for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y']:
                        try:
                            dt = datetime.strptime(date_str.split()[0], fmt)
                            parsed_publish_date = dt.strftime('%Y-%m-%d')
                            break
                        except:
                            continue
                except Exception:
                    pass  # If parsing fails, leave as None
            
            return {
                'url': url,
                'title': title,
                'content': content,
                'source': domain,
                'publish_date': parsed_publish_date
            }
            
        except Exception as e:
            raise ValueError(f"Error scraping {url} with Firecrawl: {e}") from e
    
    def load_resource_content(
        self,
        url: str,
        source_type: str = 'article',
        title: Optional[str] = None,
        upload_date: Optional[str] = None,
        publish_date: Optional[str] = None,
        resource_id: Optional[str] = None,
        **kwargs
    ) -> List[Document]:
        """
        Load and chunk article content from a URL.
        
        Args:
            url: Article URL
            source_type: Source type identifier from YAML top-level key (e.g., 'article')
            title: Article title (optional, will be scraped if not provided)
            upload_date: Upload date in YYYY-MM-DD format (optional, for metadata)
            publish_date: Publish date in YYYY-MM-DD format (optional, for metadata)
            resource_id: Unique resource identifier (optional, should be provided in YAML for articles)
            **kwargs: Additional arguments (unused for articles)
            
        Returns:
            List of Document objects with article chunks and enhanced metadata
        """
        # Scrape article content
        article_data = self.scrape_article_content(url)
        
        # Use scraped title if not provided
        if not title:
            title = article_data['title']
        
        # Auto-generate resource_id from URL if not provided
        if not resource_id:
            # Generate a simple hash-based ID from URL
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            resource_id = f"art{url_hash}"
        
        # Use extracted publish_date if not provided
        extracted_publish_date = article_data.get('publish_date')
        final_publish_date = publish_date or extracted_publish_date
        final_upload_date = upload_date or extracted_publish_date
        
        # Normalize URL
        url_normalized = self.normalize_url(url)
        
        # Create initial document
        initial_doc = Document(
            page_content=article_data['content'],
            metadata={
                'source': url_normalized,
                'title': title,
                'source_domain': article_data['source']
            }
        )
        
        # Split document into chunks
        docs = self.text_splitter.split_documents([initial_doc])
        
        # Enhance metadata for all chunks
        for doc in docs:
            doc.metadata['source_type'] = source_type
            doc.metadata['url'] = url_normalized
            
            if resource_id:
                doc.metadata['resource_id'] = resource_id
            
            if title:
                doc.metadata['title'] = title
            
            if final_upload_date:
                doc.metadata['upload_date'] = self._ensure_string_date(final_upload_date)
            
            if final_publish_date:
                doc.metadata['publish_date'] = self._ensure_string_date(final_publish_date)
            elif final_upload_date:
                doc.metadata['publish_date'] = self._ensure_string_date(final_upload_date)
        
        return docs
    
    def add_resource_to_vectorstore(
        self,
        url: str,
        source_type: str = 'article',
        title: Optional[str] = None,
        upload_date: Optional[str] = None,
        publish_date: Optional[str] = None,
        resource_id: Optional[str] = None,
        **kwargs
    ) -> int:
        """
        Add a single article to the vectorstore.
        
        Args:
            url: Article URL
            source_type: Source type identifier from YAML top-level key (e.g., 'article')
            title: Article title (optional, will be scraped if not provided)
            upload_date: Upload date in YYYY-MM-DD format (optional, for metadata)
            publish_date: Publish date in YYYY-MM-DD format (optional, for metadata)
            resource_id: Unique resource identifier (optional, should be provided in YAML for articles)
            **kwargs: Additional arguments (unused for articles)
            
        Returns:
            Number of document chunks added
        """
        # Load article content with enhanced metadata
        docs = self.load_resource_content(
            url,
            source_type=source_type,
            title=title,
            upload_date=upload_date,
            publish_date=publish_date,
            resource_id=resource_id,
            **kwargs
        )
        
        if not docs:
            logger.warning(f"No content found for {url}")
            return 0
        
        # Add documents to vectorstore using base class method
        self._add_documents_to_vectorstore(docs)
        
        return len(docs)
    
    def update_vectorstore_from_yaml(
        self,
        vectordata_yaml: Optional[Path] = None,
        skip_existing: bool = True
    ) -> Dict[str, int]:
        """
        Update vectorstore with all articles from YAML file.
        
        Args:
            vectordata_yaml: Path to YAML file (uses self.vectordata_yaml if not provided)
            skip_existing: Skip articles that are already in the vectorstore (unless force_update is True)
            
        Returns:
            Dictionary with 'added', 'skipped', 'failed', 'updated', 'chunks_added', 'chunks_deleted' counts
        """
        return super().update_vectorstore_from_yaml(
            vectordata_yaml=vectordata_yaml,
            skip_existing=skip_existing
        )
