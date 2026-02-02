#!/bin/bash
# Redeploy ClutchAI to Google Cloud Run
# Run from project root: ./scripts/gcloud/deploy.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "Deploying ClutchAI to Cloud Run..."
gcloud run deploy clutchai --source . --region us-central1 --allow-unauthenticated

echo "Deploy complete. Service URL:"
gcloud run services describe clutchai --region us-central1 --format='value(status.url)'
