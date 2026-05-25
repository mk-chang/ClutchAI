# GCP Configuration & Deployment Notes

> **Status: DECOMMISSIONED.** Migrated to Railway (2026-05-25). Cloud Run deleted. Cloud SQL + Secret Manager pending Console cleanup.

## Project

| Key | Value |
|-----|-------|
| Project ID | `clutchai-480619` |
| Region | `us-central1` |

## Cleanup Remaining (GCP Console)

- **Cloud SQL** `clutchai-db`: SUSPENDED, deletion protection enabled — disable protection → delete
- **Secret Manager**: OPENAI_API_KEY, YAHOO_CLIENT_ID, YAHOO_CLIENT_SECRET, CLOUDSQL_PASSWORD, YAHOO_ACCESS_TOKEN_JSON — billing disabled so CLI blocked; use Console
- **Artifact Registry**: Docker images from Cloud Run source deploys — delete via Console

## Former Resources (now deleted)

- Cloud Run service `clutchai` — deleted 2026-05-25
- Production URL was `https://clutchai-4mngn4dsaa-uc.a.run.app`

## Notes

- Cloud SQL suspended instance can't be patched via CLI (HTTP 409) — must use GCP Console
- Secret Manager API blocked when billing is disabled on project — use Console to delete secrets
