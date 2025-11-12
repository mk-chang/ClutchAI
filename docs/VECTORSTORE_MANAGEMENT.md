# Vectorstore Management Guide

This guide explains how to add resources (YouTube videos, articles, etc.) to your ChromaDB vectorstore.

## Overview

The `VectorstoreManager` class provides a simple way to:
1. Store resources (YouTube videos, articles, etc.) in a YAML configuration file
2. Add resources to your ChromaDB vectorstore
3. Track which resources have been added
4. Update the vectorstore with new resources from the YAML file

## Quick Start

### 1. Add Resources to Your Configuration

Edit the YAML file to add your resources.

#### Manually (Edit YAML file):

Resources are stored in `ClutchAI/rag/vectordata.yaml`. You can edit this file directly:

```yaml
youtube:
  - title: "Your Video Title"
    url: "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
    resource_id: "ytb000001"
    upload_date: "2024-01-01"
    publish_date: "2024-01-01"
    force_update: false
    remove: false

article:
  - title: "Your Article Title"
    url: "https://example.com/article"
    resource_id: "art000001"
    upload_date: "2024-01-01"
    publish_date: "2024-01-01"
    force_update: false
    remove: false
```

### 2. Add Resources to Vectorstore

Once you have resources in your YAML file, add them to the vectorstore:

```python
from ClutchAI.rag.vectorstore import VectorstoreManager
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

# Initialize manager
manager = VectorstoreManager()

# Add a single video to vectorstore
chunks = manager.add_video_to_vectorstore(
    url="https://www.youtube.com/watch?v=TB2QwCRMams",
    chunk_size_seconds=30,
    source_type="youtube"
)
print(f"Added {chunks} chunks")

# Add a single article to vectorstore
chunks = manager.add_article_to_vectorstore(
    url="https://example.com/article",
    source_type="article"
)
print(f"Added {chunks} chunks")

# Or update vectorstore with all resources from YAML file
results = manager.update_vectorstore(
    chunk_size_seconds=30,
    skip_existing=True  # Skip resources already in vectorstore
)
print(f"Added: {results['added']}, Skipped: {results['skipped']}, Failed: {results['failed']}")
```

### 3. Check Status

```python
# Get vectorstore statistics
stats = manager.get_vectorstore_stats()
print(f"Vectorstore exists: {stats['exists']}")
print(f"Document count: {stats['document_count']}")
print(f"YouTube URLs: {stats['youtube_urls']}")
print(f"Article URLs: {stats['article_urls']}")
print(f"Total URLs in vectorstore: {stats['urls_in_vectorstore']}")
```

## Complete Example

```python
from ClutchAI.rag.vectorstore import VectorstoreManager
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

# Initialize manager (will use ClutchAI/rag/vectordata.yaml by default)
manager = VectorstoreManager()

# 1. Edit ClutchAI/rag/vectordata.yaml to add your resources
#    (See example YAML structure above)

# 2. Update vectorstore with all resources from YAML file
results = manager.update_vectorstore(
    chunk_size_seconds=30,
    skip_existing=True
)

print(f"\nUpdate Results:")
print(f"  - Added: {results['added']} resources")
print(f"  - Skipped: {results['skipped']} resources (already in vectorstore)")
print(f"  - Failed: {results['failed']} resources")
print(f"  - Total chunks: {results['chunks_added']}")

# 3. Check final status
stats = manager.get_vectorstore_stats()
print(f"\nFinal Stats:")
print(f"  - YouTube URLs: {stats['youtube_urls']}")
print(f"  - Article URLs: {stats['article_urls']}")
print(f"  - Documents in vectorstore: {stats['document_count']}")
```

## File Locations

- **Resources configuration**: `ClutchAI/rag/vectordata.yaml`
- **Vectorstore**: `ClutchAI/rag/chroma_db/` (created automatically by VectorstoreManager)
- **Default location**: When using `ClutchAIAgent`, the vectorstore is automatically initialized at `ClutchAI/rag/chroma_db/`

## Notes

1. **Transcripts**: YouTube videos must have captions/transcripts available. Private videos or videos without transcripts will fail.

2. **Chunk Size**: The `chunk_size_seconds` parameter controls how YouTube transcripts are split. Smaller chunks (e.g., 30 seconds) provide more granular retrieval, while larger chunks (e.g., 60 seconds) provide more context. Articles are split using a different strategy based on text content.

3. **Duplicate Prevention**: The manager automatically skips resources that are already in the vectorstore when `skip_existing=True`. Resources are identified by their URL.

4. **Persistence**: The vectorstore is automatically persisted to disk at `ClutchAI/rag/chroma_db/`, so your data persists between runs.

5. **Metadata**: Each document in the vectorstore includes metadata:
   - `url`: The resource URL (YouTube video or article)
   - `source_type`: Type of resource (e.g., "youtube", "article")
   - For YouTube videos: `start_seconds`, `start_timestamp` for chunk timing
   - Additional resource info if available (title, publish_date, etc.)

6. **YAML Structure**: Resources are organized by type (youtube, article) in the YAML file. Each resource requires:
   - `title`: Resource title
   - `url`: Resource URL
   - `resource_id`: Unique identifier for the resource
   - `upload_date`: Date when resource was added (YYYY-MM-DD format)
   - `publish_date`: Original publish date (YYYY-MM-DD format)
   - `force_update`: Set to `true` to force re-processing
   - `remove`: Set to `true` to remove from vectorstore

## Troubleshooting

### "Failed to load transcript"
- The video may not have captions/transcripts available
- The video may be private or restricted
- Check the video URL is correct

### "Vectorstore not found"
- The vectorstore will be created automatically when you add the first resource
- Ensure you have write permissions to the `ClutchAI/rag/` directory
- The vectorstore location defaults to `ClutchAI/rag/chroma_db/` when using `ClutchAIAgent`

### "OpenAI API key is required"
- Set the `OPENAI_API_KEY` environment variable
- Or pass `openai_api_key` parameter when initializing `VectorstoreManager`

## Integration with ClutchAI Agent

The vectorstore created by `VectorstoreManager` is compatible with `ClutchAIAgent`. The agent automatically:
- Initializes the vectorstore at `ClutchAI/rag/chroma_db/` when created
- Updates the vectorstore with resources from `ClutchAI/rag/vectordata.yaml` on initialization
- Uses the vectorstore for RAG queries when answering questions about your fantasy league

You don't need to manually manage the vectorstore when using `ClutchAIAgent` - it handles everything automatically. Just make sure your `ClutchAI/rag/vectordata.yaml` file is up to date with the resources you want to include.



