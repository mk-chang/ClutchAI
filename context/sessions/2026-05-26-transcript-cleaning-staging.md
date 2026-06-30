# Session — 2026-05-26

## What We Did

### 1. Transcript Cleaning Feature (`_clean_documents`)

Implemented LLM-based transcript/article cleaning before vectorstore ingestion. Uses `gpt-4o-mini` to identify and remove ads, intros, outros, sponsor reads, and boilerplate chunks.

**Architecture (template method pattern):**
- `BaseVectorManager._run_cleaning(docs, prompt, model)` — shared LLM mechanics (call, parse JSON, filter, fallback on error)
- `BaseVectorManager._clean_documents(docs)` — abstract; each subclass implements with domain-specific prompt
- `YoutubeVectorManager._clean_documents()` — podcast/transcript prompt (ads, sponsor reads, social plugs)
- `ArticleVectorManager._clean_documents()` — article prompt (nav, cookie notices, newsletter prompts, paywalls)

**Wired into pipeline:**
- `YoutubeVectorManager.load_resource_content()` — calls `_clean_documents()` after transcript fetch, before metadata enhancement
- `ArticleVectorManager.load_resource_content()` — calls `_clean_documents()` after chunking

**Logging (always-on):**
```
Cleaned: 42 → 39 chunks | 8,432 → 7,891 chars | removed 3
```

**DEV_MODE logging (when `DEV_MODE=true`):**
- Before: first 3 chunk previews (150 chars each)
- After: each removed chunk preview

**Tests:** 8 tests in `tests/test_youtube_vector_manager.py` — all passing.

### 2. Staging Environment Setup

Created Railway `staging` environment (forked from production by user):
- `ClutchAI` service: `OPENAI_MODEL=gpt-4o-mini`, `DISABLE_RAG=true`
- `lockedon-cron`: schedule changed to `0 0 1 1 *` (once/year = effectively disabled)
- `staging` git branch created and pushed

**One manual step remaining:** In Railway dashboard → staging → ClutchAI service → Settings → Source → change branch from `main` to `staging`. Same for `lockedon-cron`.

**Workflow going forward:**
```
feature/X → staging (test) → main (production)
```

### 3. DEV_MODE + Staging Cron Config

- `DEV_MODE=true` in staging → verbose before/after logging + max 3 videos per cron run
- `DEV_MODE=false` in production (default) → summary line only, full 15 videos per run
- Cap logic in `scripts/pipelines/update_lockedon_knowledge.py`: `if DEV_MODE: max_videos_added = 3`

### 4. Railway Token / Auth Fix

- Added `RAILWAY_TOKEN` to `~/.claude/settings.json` `env` block → MCP server inherits it
- Added to `~/.zshrc` for terminal usage
- Eliminates `railway login` + VS Code reload requirement

### 5. Branch Cleanup

- `railway_migration` branch deleted (local + remote) — all commits already in `main` via rebase
- `feature/podcast_cleanup` left as-is (only contains planning docs, no code)

## Files Changed

| File | Change |
|------|--------|
| `data/postgres/vector_managers/base.py` | Added `_run_cleaning()` helper + abstract `_clean_documents()` |
| `data/postgres/vector_managers/youtube.py` | Implemented `_clean_documents()` with podcast prompt |
| `data/postgres/vector_managers/article.py` | Implemented `_clean_documents()` with article prompt + wired into `load_resource_content` |
| `scripts/pipelines/update_lockedon_knowledge.py` | Cap `max_videos_added=3` when `DEV_MODE=true` |
| `tests/test_youtube_vector_manager.py` | 8 tests for `_clean_documents` and wired pipeline |
| `~/.claude/settings.json` | Added `RAILWAY_TOKEN` to `env` block |
| `~/.zshrc` | Added `export RAILWAY_TOKEN=...` |

## Commits This Session

```
33cbb12 feat: add _clean_documents to BaseVectorManager with gpt-4o-mini filtering
8157be9 feat: clean YouTube transcript chunks before vectorstore ingestion
4bfeda5 refactor: make _clean_documents abstract with subclass-specific prompts
482a7d5 feat: add DEV_MODE logging to _run_cleaning for before/after inspection
1fa4032 feat: cap max_videos_added at 3 in DEV_MODE for faster staging runs
ccd08cc feat: add DEV_MODE logging to _run_cleaning for before/after inspection
531925a feat: always log chunk/char counts after cleaning, DEV_MODE for samples only
```

## State Left In

- All 7 commits on `main` locally, **not yet pushed to origin/main**
- `staging` branch is up to date (has all 7 commits)
- Tests: 8/8 passing + 5/5 supadata tests passing
- Railway staging: needs branch source changed to `staging` in dashboard (manual)
- Railway MCP: needs VS Code reload to pick up `RAILWAY_TOKEN` from settings.json

## Notes

- `_run_cleaning` uses module-level `from openai import OpenAI` in base.py (required for `patch('data.postgres.vector_managers.base.OpenAI')` in tests to work — local import inside method breaks the patch target)
- Article cleaning runs on post-chunked docs; YouTube cleaning runs on raw loader output — both before metadata enhancement
- `lockedon-cron` in staging set to `0 0 1 1 *` (Jan 1 midnight) not removed — Railway services are project-level, can't be removed from a single environment
