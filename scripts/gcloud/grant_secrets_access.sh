#!/bin/bash
#
# Grant the Cloud Run service account permission to read secrets from Secret Manager.
# Run this once before deploying, or if you get "Permission denied" on secret access.
#
# Usage: ./scripts/gcloud/grant_secrets_access.sh [project_id]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECT_ID="${1:-clutchai-480619}"

# Try to get project from .env if not passed
if [[ -f "$PROJECT_ROOT/.env" ]]; then
  ENV_PROJECT=$(grep -E "^GOOGLE_CLOUD_PROJECT=" "$PROJECT_ROOT/.env" 2>/dev/null | cut -d'=' -f2- | tr -d '"' | tr -d "'" | xargs)
  [[ -n "$ENV_PROJECT" ]] && PROJECT_ID="$ENV_PROJECT"
fi

echo "Granting Secret Manager access to Cloud Run service account..."
echo "Project: $PROJECT_ID"
echo ""

PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
SERVICE_ACCOUNT="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/secretmanager.secretAccessor" \
  --quiet

echo ""
echo "Done. The Cloud Run service account can now read secrets from Secret Manager."
