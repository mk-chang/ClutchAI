"""
Master script to update the vector database from all sources.

This script runs all vectordb pipeline scripts in sequence:
1. update_base_knowledge.py - Updates from knowledge_base.yaml (manual YouTube videos and articles)
2. update_lockedon_knowledge.py - Updates LockedOn Fantasy Basketball podcast episodes

Usage:
    python scripts/update_vector_database.py
"""

import sys
import traceback
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add project root to path
# From scripts/update_vector_database.py
# .parent = scripts/
# .parent.parent = project root
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load environment variables
env_path = project_root / '.env'
load_dotenv(env_path)

# Import logger
from logger import get_logger, setup_logging

# Import the pipeline scripts
from scripts.vectordb_pipelines.update_base_knowledge import main as update_base_knowledge
from scripts.vectordb_pipelines.update_lockedon_knowledge import main as update_lockedon_knowledge

# Setup logging
setup_logging(debug=False)
logger = get_logger(__name__)


def run_pipeline(name, pipeline_func):
    """Run a single pipeline script and handle errors."""
    logger.info("\n" + "=" * 80)
    logger.info(f"Running: {name}")
    logger.info("=" * 80)
    
    pipeline_start_time = datetime.now()
    try:
        pipeline_func()
        pipeline_end_time = datetime.now()
        pipeline_duration = pipeline_end_time - pipeline_start_time
        logger.info(f"\n✅ {name} completed successfully")
        return True, pipeline_duration
    except Exception as e:
        pipeline_end_time = datetime.now()
        pipeline_duration = pipeline_end_time - pipeline_start_time
        logger.error(f"\n❌ {name} failed with error:")
        logger.error(f"   {str(e)}")
        logger.error("\nTraceback:")
        traceback.print_exc()
        return False, pipeline_duration


def main():
    """Run all vector database update pipelines."""
    start_time = datetime.now()
    
    logger.info("=" * 80)
    logger.info("Vector Database Update Pipeline - Master Script")
    logger.info("=" * 80)
    logger.info(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    # Define all pipelines to run
    pipelines = [
        ("Base Knowledge Update (knowledge_base.yaml)", update_base_knowledge),
        ("LockedOn Knowledge Update", update_lockedon_knowledge),
    ]
    
    # Track results and durations
    results = {}
    durations = {}
    
    # Run each pipeline
    for name, pipeline_func in pipelines:
        success, pipeline_duration = run_pipeline(name, pipeline_func)
        results[name] = success
        durations[name] = pipeline_duration
        
        # Optionally stop on first failure (uncomment if desired)
        # if not success:
        #     logger.warning("\n⚠️  Stopping pipeline execution due to failure.")
        #     break
    
    # Calculate summary
    end_time = datetime.now()
    total_duration = end_time - start_time
    
    # Log summary
    logger.info("\n" + "=" * 80)
    logger.info("Pipeline Execution Summary")
    logger.info("=" * 80)
    for name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"{status}: {name}")
    logger.info("=" * 80)
    logger.info(f"Total duration: {total_duration}")
    logger.info(f"Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    # Print duration breakdown at the end
    logger.info("\n" + "=" * 80)
    logger.info("Duration Breakdown")
    logger.info("=" * 80)
    for name, pipeline_duration in durations.items():
        # Format duration
        total_seconds = pipeline_duration.total_seconds()
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        if hours > 0:
            duration_str = f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            duration_str = f"{minutes}:{seconds:02d}"
        logger.info(f"  {name}: {duration_str}")
    
    # Format total duration
    total_seconds = total_duration.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    if hours > 0:
        total_duration_str = f"{hours}:{minutes:02d}:{seconds:02d}"
    else:
        total_duration_str = f"{minutes}:{seconds:02d}"
    logger.info(f"  Total: {total_duration_str}")
    logger.info("=" * 80)
    
    # Exit with error code if any pipeline failed
    if not all(results.values()):
        logger.warning("\n⚠️  Some pipelines failed. Check the output above for details.")
        sys.exit(1)
    else:
        logger.info("\n✅ All pipelines completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()

