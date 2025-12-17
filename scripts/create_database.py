#!/usr/bin/env python3
"""
Script to create the ClutchAI database in Google Cloud SQL.

This script connects to the default 'postgres' database and creates
the database specified in CLOUDSQL_DATABASE environment variable.

Usage:
    python scripts/create_database.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.cloud_sql.connection import create_database_if_not_exists
from logger import get_logger, setup_logging

logger = get_logger(__name__)


def main():
    """Create the database if it doesn't exist."""
    # Setup logging
    setup_logging()
    
    # Load environment variables from .env file if it exists
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    
    logger.info("Creating ClutchAI database...")
    logger.info(f"Database: {os.environ.get('CLOUDSQL_DATABASE', 'Not set')}")
    logger.info(f"Instance: {os.environ.get('CLOUDSQL_INSTANCE', 'Not set')}")
    logger.info(f"Region: {os.environ.get('CLOUDSQL_REGION', 'Not set')}")
    
    success = create_database_if_not_exists()
    
    if success:
        logger.info("Database setup complete!")
        logger.info("You can now run your vectordb pipelines.")
        return 0
    else:
        logger.error("Failed to create database. Please check the error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

