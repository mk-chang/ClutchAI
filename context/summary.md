# Project Summary

_High-level state of the project. Update when significant milestones are reached._

## Current State

Multi-agent fantasy basketball assistant fully deployed on Railway with a staging environment. Transcript/article cleaning is now active in the pipeline — `gpt-4o-mini` filters ads, intros, and boilerplate from ingested content before vectorstore storage. Streamlit frontend live at `https://clutchai-production.up.railway.app`.

## What's Working

- Multi-agent pipeline (Supervisor → Research agents → Analysis)
- Yahoo Fantasy API integration (45 tools)
- NBA API integration (16 tools)
- pgvector knowledge base on Railway PostgreSQL (YouTube transcripts + articles)
- RAG retrieval active (DISABLE_RAG=false in production)
- Streamlit UI
- Railway deployment: `main` → production, `staging` → staging (branch source pending manual dashboard config)
- Hourly cron job (`lockedon-cron`) populating LockedOn Fantasy Basketball podcast transcripts
- LLM-based transcript/article cleaning (`_clean_documents`) on all ingested content
- DEV_MODE for verbose pipeline logging + reduced batch size (3 videos) in staging

## What's In Progress / Incomplete

- **LockedOn backlog**: 655 YouTube videos ingesting via hourly cron (~15/run)
- **Staging branch source**: Needs manual Railway dashboard config (ClutchAI + lockedon-cron → `staging` branch)
- **GCP teardown**: Cloud Run deleted; Cloud SQL + Secret Manager still need manual Console cleanup
- **Waiver wire tool**: Implemented with Postgres-persisted cache on `feature/waiver_wire` branch (not yet merged to main)
- **Player stats database**: Design in progress. Schema (4 tables) and pipeline (nightly cron) approved. Agent tools section pending. See `context/sessions/2026-06-29-player-stats-db-brainstorm.md`.

## Architecture Notes

See `CLAUDE.md` for full architecture overview and `context/agents/patterns.md` for agent-specific notes.

| What | Path |
|------|------|
| DB layer | `data/postgres/` |
| Pipeline scripts | `scripts/pipelines/` |
| Vector managers | `data/postgres/vector_managers/` |
| Cron master script | `scripts/pipelines/update_vector_database.py` |
| Cleaning base | `data/postgres/vector_managers/base.py` → `_run_cleaning()` |
