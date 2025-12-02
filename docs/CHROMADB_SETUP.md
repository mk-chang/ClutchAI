# ChromaDB Setup Guide for ClutchAI

## Quick Setup

### 1. Install Dependencies

**Install all dependencies:**
```bash
pip install -r requirements.txt
```

**Or install ChromaDB only (minimal setup):**
```bash
pip install chromadb langchain langchain-openai langchain-community
```

### 2. Configure Environment

Create a `.env` file with these settings:

```bash
# Copy the configuration template
cp env.example .env

# Edit .env file and update these values:
# - YAHOO_CLIENT_ID=your_yahoo_client_id
# - YAHOO_CLIENT_SECRET=your_yahoo_client_secret  
# - OPENAI_API_KEY=your_openai_api_key
# - YAHOO_LEAGUE_ID=your_league_id (optional, defaults to 58930)
```

## ChromaDB Storage Location

ChromaDB will automatically create a local storage directory:

- **Location**: `ClutchAI/rag/chroma_db/` (in the rag directory)
- **Persistence**: Automatic - data persists between runs
- **Backup**: Simply copy the `ClutchAI/rag/chroma_db/` directory
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
   pip install chromadb langchain langchain-openai langchain-community
   ```

2. **Environment Variables**: Check your `.env` file has `OPENAI_API_KEY` set
   ```bash
   OPENAI_API_KEY=your_openai_api_key
   ```

3. **Storage Permissions**: Ensure write access to project directory
   ```bash
   chmod 755 ClutchAI/rag/chroma_db/
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

1. **Start the Streamlit application**:
   ```bash
   streamlit run app/streamlit_app.py
   ```

2. **Test the application**:
   - Open http://localhost:8501
   - Enter your API keys in the sidebar
   - Ask questions about your fantasy league
   - The vectorstore will be automatically initialized and used for RAG queries

3. **Monitor storage**:
   - Check `ClutchAI/rag/chroma_db/` directory
   - Files will be created automatically when you add resources

## Production Considerations

For production use with ChromaDB:

- **Backup Strategy**: Regular backups of `ClutchAI/rag/chroma_db/` directory
- **Storage Space**: Monitor disk usage
- **Performance**: Consider upgrading to cloud solutions for scale
- **Security**: Ensure proper file permissions

## Adding Resources to Vectorstore

To add YouTube videos or articles to your vectorstore, edit `ClutchAI/rag/vector_data.yaml` and then use the `VectorstoreManager`:

```python
from ClutchAI.rag.vectorstore import VectorstoreManager

manager = VectorstoreManager()
results = manager.update_vectorstore()
```

See `docs/VECTORSTORE_MANAGEMENT.md` for detailed instructions.
