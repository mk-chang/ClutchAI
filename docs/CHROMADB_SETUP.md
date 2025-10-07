# ChromaDB Setup Guide for ClutchAI

## Quick Setup

### 1. Install Dependencies

**Install all dependencies:**
```bash
pip install -r requirements.txt
```

**Or install ChromaDB only (minimal setup):**
```bash
pip install chromadb fastapi uvicorn pydantic pydantic-settings
```

### 2. Configure Environment

Create a `.env` file with these settings:

```bash
# Copy the configuration
cp chromadb.env .env

# Edit .env file and update these values:
# - YAHOO_CLIENT_ID=your_yahoo_client_id
# - YAHOO_CLIENT_SECRET=your_yahoo_client_secret  
# - OPENAI_API_KEY=your_openai_api_key
# - DATABASE_URL=postgresql://username:password@localhost:5432/clutchai
# - DATABASE_URL_ASYNC=postgresql+asyncpg://username:password@localhost:5432/clutchai
```

### 3. Set Vector Database Provider

Make sure your `.env` file contains:
```bash
VECTOR_DB_PROVIDER=chromadb
```

### 4. Test ChromaDB Setup

```bash
python scripts/setup_chromadb.py
```

## ChromaDB Storage Location

ChromaDB will automatically create a local storage directory:

- **Location**: `./chroma_db/` (in your project root)
- **Persistence**: Automatic - data persists between runs
- **Backup**: Simply copy the `./chroma_db/` directory
- **Portability**: Easy to move between environments

## Benefits of ChromaDB

✅ **Local Storage**: No external dependencies  
✅ **Easy Setup**: No API keys or external services  
✅ **Automatic Persistence**: Data saved locally  
✅ **Cost Effective**: Free for local development  
✅ **Easy Backup**: Just copy the directory  
✅ **Fast**: Optimized for local operations  

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed
   ```bash
   pip install chromadb fastapi pydantic
   ```

2. **Environment Variables**: Check your `.env` file
   ```bash
   VECTOR_DB_PROVIDER=chromadb
   ```

3. **Storage Permissions**: Ensure write access to project directory
   ```bash
   chmod 755 ./chroma_db/
   ```

### Quick Test

Test ChromaDB directly:

```python
import chromadb

# Test ChromaDB
client = chromadb.Client()
collection = client.create_collection("test")
collection.add(
    documents=["This is a test document"],
    ids=["1"]
)
results = collection.query(query_texts=["test"], n_results=1)
print("ChromaDB working:", len(results['documents'][0]) > 0)
```

## Next Steps

1. **Start the application**:
   ```bash
   uvicorn app.main:app --reload
   ```

2. **Test the API**:
   - Open http://localhost:8000/docs
   - Try the vector endpoints
   - Test RAG queries

3. **Monitor storage**:
   - Check `./chroma_db/` directory
   - Files will be created automatically

## Production Considerations

For production use with ChromaDB:

- **Backup Strategy**: Regular backups of `./chroma_db/` directory
- **Storage Space**: Monitor disk usage
- **Performance**: Consider upgrading to cloud solutions for scale
- **Security**: Ensure proper file permissions

## Alternative Vector Databases

If you need cloud storage later:

- **Pinecone**: Cloud-hosted, managed service
- **Qdrant**: Self-hosted, high performance
- **Weaviate**: Graph database with vector search

Just change `VECTOR_DB_PROVIDER` in your `.env` file!
