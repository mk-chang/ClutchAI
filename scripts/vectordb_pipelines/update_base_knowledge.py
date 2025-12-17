"""
Simple script to update vectordb from YAML file.

Usage:
    python scripts/vectordb_pipelines/update_base_knowledge.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
# From scripts/vectordb_pipelines/update_base_knowledge.py
# .parent = scripts/vectordb_pipelines/
# .parent.parent = scripts/
# .parent.parent.parent = project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data.cloud_sql.connection import PostgresConnection
from data.cloud_sql.vector_managers import YoutubeVectorManager, ArticleVectorManager
from langchain_openai import OpenAIEmbeddings
from logger import get_logger, setup_logging

# Load environment variables
env_path = project_root / '.env'
load_dotenv(env_path)

# Setup logging
setup_logging(debug=False)
logger = get_logger(__name__)

def main():
    # Path to YAML file
    yaml_path = project_root / 'data' / 'knowledge_base.yaml'
    
    if not yaml_path.exists():
        logger.error(f"YAML file not found: {yaml_path}")
        return
    
    logger.info("Updating vectordb from YAML...")
    logger.info(f"YAML file: {yaml_path}\n")
    
    # Connect
    connection = PostgresConnection()
    embeddings = OpenAIEmbeddings(api_key=os.environ.get('OPENAI_API_KEY'))
    
    # Process YouTube videos
    logger.info("Processing YouTube videos...")
    youtube_manager = YoutubeVectorManager(
        connection=connection,
        embeddings=embeddings,
        table_name=None,  # Uses CLOUDSQL_VECTOR_TABLE from env
        vectordata_yaml=yaml_path,
    )
    youtube_results = youtube_manager.update_vectorstore_from_yaml(
        vectordata_yaml=yaml_path,
        skip_existing=True
    )
    
    # Process articles
    logger.info("\nProcessing articles...")
    article_manager = ArticleVectorManager(
        connection=connection,
        embeddings=embeddings,
        table_name=None,  # Uses CLOUDSQL_VECTOR_TABLE from env
        vectordata_yaml=yaml_path,
    )
    article_results = article_manager.update_vectorstore_from_yaml(
        vectordata_yaml=yaml_path,
        skip_existing=True
    )
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"YouTube - Added: {youtube_results.get('added', 0)}, "
          f"Skipped: {youtube_results.get('skipped', 0)}, "
          f"Failed: {youtube_results.get('failed', 0)}")
    logger.info(f"Articles - Added: {article_results.get('added', 0)}, "
          f"Skipped: {article_results.get('skipped', 0)}, "
          f"Failed: {article_results.get('failed', 0)}")
    logger.info(f"Total chunks added: {youtube_results.get('chunks_added', 0) + article_results.get('chunks_added', 0):,}")
    logger.info("=" * 60)
    
    connection.close()
    logger.info("\n✅ Update complete!")

if __name__ == "__main__":
    main()

