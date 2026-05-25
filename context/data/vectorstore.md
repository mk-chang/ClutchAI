# Vectorstore Notes

## Schema

| Table | Purpose |
|-------|---------|
| `vectorstore` | pgvector embeddings (YouTube transcripts + articles) |
| `app` | App-level data |

See `data/postgres/schema.py` for full schema. Vector managers: `data/postgres/vector_managers/youtube.py`, `article.py`.

## Data Sources

| Source | Pipeline Script | Config |
|--------|----------------|--------|
| LockedOn NBA podcasts (YouTube) | `scripts/pipelines/update_lockedon_knowledge.py` | `config/vector_config.yaml` |
| Articles | `scripts/pipelines/update_vector_database.py` | `config/vector_config.yaml` |

Knowledge base URLs/channels defined in `data/knowledge_base.yaml`.

## RAG Settings (`config/rag_config.yaml`)

- `youtube.chunk_size_seconds` — transcript chunk size
- `vectorstore.k` — number of documents to retrieve
- `retrieval.search_type` — `"similarity"` or `"mmr"`

## Pipeline Settings (`config/vector_config.yaml`)

- `youtube_channel.max_videos` — max videos processed per run
- `youtube_channel.delay_between_videos` — delay to avoid YouTube rate limiting

## Pipeline Run Notes

_Update this section after running pipeline updates: date, videos/articles added, any issues._
