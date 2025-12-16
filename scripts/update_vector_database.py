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

# Import the pipeline scripts
from scripts.vectordb_pipelines.update_base_knowledge import main as update_base_knowledge
from scripts.vectordb_pipelines.update_lockedon_knowledge import main as update_lockedon_knowledge


def run_pipeline(name, pipeline_func):
    """Run a single pipeline script and handle errors."""
    print("\n" + "=" * 80)
    print(f"Running: {name}")
    print("=" * 80)
    
    try:
        pipeline_func()
        print(f"\n✅ {name} completed successfully")
        return True
    except Exception as e:
        print(f"\n❌ {name} failed with error:")
        print(f"   {str(e)}")
        print("\nTraceback:")
        traceback.print_exc()
        return False


def main():
    """Run all vector database update pipelines."""
    start_time = datetime.now()
    
    print("=" * 80)
    print("Vector Database Update Pipeline - Master Script")
    print("=" * 80)
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Define all pipelines to run
    pipelines = [
        ("Base Knowledge Update (knowledge_base.yaml)", update_base_knowledge),
        ("LockedOn Knowledge Update", update_lockedon_knowledge),
    ]
    
    # Track results
    results = {}
    
    # Run each pipeline
    for name, pipeline_func in pipelines:
        success = run_pipeline(name, pipeline_func)
        results[name] = success
        
        # Optionally stop on first failure (uncomment if desired)
        # if not success:
        #     print("\n⚠️  Stopping pipeline execution due to failure.")
        #     break
    
    # Print summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 80)
    print("Pipeline Execution Summary")
    print("=" * 80)
    for name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status}: {name}")
    print("=" * 80)
    print(f"Total duration: {duration}")
    print(f"Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Exit with error code if any pipeline failed
    if not all(results.values()):
        print("\n⚠️  Some pipelines failed. Check the output above for details.")
        sys.exit(1)
    else:
        print("\n✅ All pipelines completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()

