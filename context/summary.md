# Project Summary

_High-level state of the project. Update when significant milestones are reached._

## Current State

Multi-agent fantasy basketball assistant. Core multi-agent system is built and working: Supervisor routes queries to Yahoo Fantasy, Statistic, and News agents, then passes results to the Analysis agent. Streamlit frontend deployed on Google Cloud Run.

## What's Working

- Multi-agent pipeline (Supervisor → Research agents → Analysis)
- Yahoo Fantasy API integration (45 tools)
- NBA API integration (16 tools)
- pgvector knowledge base on Google Cloud SQL (YouTube transcripts + articles)
- Streamlit UI
- Google Cloud Run deployment

## What's In Progress / Incomplete

_Update as work progresses._

## Architecture Notes

See `CLAUDE.md` for full architecture overview and `context/agents/patterns.md` for agent-specific notes.
