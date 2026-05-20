# Project Summary

_High-level state of the project. Update when significant milestones are reached._

## Current State

Multi-agent fantasy basketball assistant. Core multi-agent system is built and working: Supervisor routes queries to Yahoo Fantasy, Statistic, and News agents, then passes results to the Analysis agent. Streamlit frontend deployed on Google Cloud Run (being migrated to Railway).

## What's Working

- Multi-agent pipeline (Supervisor → Research agents → Analysis)
- Yahoo Fantasy API integration (45 tools)
- NBA API integration (16 tools)
- pgvector knowledge base on Google Cloud SQL (YouTube transcripts + articles)
- Streamlit UI
- Google Cloud Run deployment (active, being replaced)

## What's In Progress / Incomplete

- **Railway migration** (`railway_migration` branch) — code changes done, infra setup pending
  - `connection.py` rewritten to use `DATABASE_URL` (psycopg2, no Cloud SQL Connector)
  - Env vars renamed: `CLOUDSQL_VECTOR_TABLE` → `VECTOR_TABLE`, `CLOUDSQL_APP_TABLE` → `APP_TABLE`
  - Railway project setup, vectorstore population, and cron job setup still needed

## Architecture Notes

See `CLAUDE.md` for full architecture overview and `context/agents/patterns.md` for agent-specific notes.
