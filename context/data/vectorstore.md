# Vectorstore Notes

## Schema

| Table | Purpose |
|-------|---------|
| `locked_on_vector` | pgvector embeddings (YouTube transcripts + articles) |
| `app` | App-level data |

See `data/postgres/schema.py` for full schema. Vector managers: `data/postgres/vector_managers/youtube.py`, `article.py`.

## Data Sources

| Source | Pipeline Script | Config |
|--------|----------------|--------|
| Static seed (knowledge_base.yaml) | `scripts/pipelines/update_base_knowledge.py` | `config/vector_config.yaml` |
| LockedOn NBA podcasts (YouTube) | `scripts/pipelines/update_lockedon_knowledge.py` | `config/vector_config.yaml` |
| Master (runs both) | `scripts/pipelines/update_vector_database.py` | — |

Knowledge base URLs/channels defined in `data/knowledge_base.yaml`.

## RAG Settings (`config/rag_config.yaml`)

- `youtube.chunk_size_seconds` — transcript chunk size
- `vectorstore.k` — number of documents to retrieve
- `retrieval.search_type` — `"similarity"` or `"mmr"`

## Pipeline Settings (`config/vector_config.yaml`)

- `youtube_channel.max_videos_added: 15` — stays just under YouTube IP-block threshold (~17 videos)
- `youtube_channel.delay_between_videos: 15.0` — delay to avoid YouTube rate limiting

## YouTube IP Blocking & Supadata Fallback

YouTube blocks IP after ~17 videos in one session. Mitigations:
- Batch size 15 (below threshold) — each cron run is a new process/session
- Circuit breaker: `_youtube_blocked` flag on `YoutubeVectorManager` — once blocked, skips YouTube retries entirely
- Supadata fallback (`SUPADATA_API_KEY`): calls `https://api.supadata.ai/v1/youtube/transcript` when YouTube is blocked
- 100 Supadata credits/month — preserved by circuit breaker + small batches

## Railway Cron Job

- Service: `lockedon-cron` (id: `4e548f10-24fc-4859-a946-f13f79f011c1`)
- Schedule: `0 * * * *` (hourly)
- Command: `python scripts/pipelines/update_vector_database.py`
- Backlog: 655 LockedOn videos (~15/run → ~44 hrs to complete from 2026-05-25)
- Switch to weekly (`0 3 * * 0`) once backlog is done

## Pipeline Run Notes

- 2026-05-25: Static knowledge (knowledge_base.yaml) loaded. LockedOn backlog ingestion started via hourly cron.
