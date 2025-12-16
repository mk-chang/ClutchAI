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


def main():
    """Create the database if it doesn't exist."""
    # Load environment variables from .env file if it exists
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    
    print("Creating ClutchAI database...")
    print(f"Database: {os.environ.get('CLOUDSQL_DATABASE', 'Not set')}")
    print(f"Instance: {os.environ.get('CLOUDSQL_INSTANCE', 'Not set')}")
    print(f"Region: {os.environ.get('CLOUDSQL_REGION', 'Not set')}")
    print()
    
    success = create_database_if_not_exists()
    
    if success:
        print("\n✓ Database setup complete!")
        print("You can now run your vectordb pipelines.")
        return 0
    else:
        print("\n✗ Failed to create database. Please check the error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

