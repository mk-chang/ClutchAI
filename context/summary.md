# Project Summary

_High-level state of the project. Update when significant milestones are reached._

## Current State

Multi-agent fantasy basketball assistant fully deployed on Railway. Supervisor routes queries to Yahoo Fantasy, Statistics, and News research agents, then synthesizes via Analysis agent. Streamlit frontend live at `https://clutchai-production.up.railway.app`.

## What's Working

- Multi-agent pipeline (Supervisor → Research agents → Analysis)
- Yahoo Fantasy API integration (45 tools)
- NBA API integration (16 tools)
- pgvector knowledge base on Railway PostgreSQL (YouTube transcripts + articles)
- RAG retrieval active (DISABLE_RAG=false)
- Streamlit UI
- Railway deployment (main branch, auto-deploy on push)
- Hourly cron job (`lockedon-cron`) populating LockedOn Fantasy Basketball podcast transcripts

## What's In Progress / Incomplete

- **LockedOn backlog**: 655 YouTube videos ingesting via hourly cron (~15/run, ~44 hrs to complete from 2026-05-25)
- **GCP teardown**: Cloud Run deleted; Cloud SQL + Secret Manager still need manual Console cleanup

## Architecture Notes

See `CLAUDE.md` for full architecture overview and `context/agents/patterns.md` for agent-specific notes.

| What | Path |
|------|------|
| DB layer | `data/postgres/` |
| Pipeline scripts | `scripts/pipelines/` |
| Vector managers | `data/postgres/vector_managers/` |
| Cron master script | `scripts/pipelines/update_vector_database.py` |
