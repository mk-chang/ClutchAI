#!/bin/bash
# Redeploy ClutchAI to Google Cloud Run
# Run from project root: ./scripts/gcloud/deploy.sh [--update-secrets]
#
# Options:
#   --update-secrets   Run update_secrets.sh first to sync .env to Secret Manager
#
# Prerequisites:
#   1. Run ./scripts/gcloud/update_secrets.sh (or use --update-secrets) to upload .env to Secret Manager
#   2. Run ./scripts/gcloud/grant_secrets_access.sh to grant Cloud Run access to secrets

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Optional: sync .env to Secret Manager before deploying
if [[ "${1:-}" == "--update-secrets" ]]; then
  echo "Updating secrets from .env..."
  "$SCRIPT_DIR/update_secrets.sh"
  echo ""
fi

# Secrets from Secret Manager (must exist - create via update_secrets.sh)
# Format: ENV_VAR=SECRET_NAME:latest
# Add or remove secrets to match what you have in Secret Manager
SECRETS="OPENAI_API_KEY=OPENAI_API_KEY:latest,YAHOO_CLIENT_ID=YAHOO_CLIENT_ID:latest,YAHOO_CLIENT_SECRET=YAHOO_CLIENT_SECRET:latest,CLOUDSQL_PASSWORD=CLOUDSQL_PASSWORD:latest"

# Required for Yahoo OAuth on Cloud Run (no interactive login). Create once after local OAuth - see docs/DEPLOYMENT.md
SECRETS="${SECRETS},YAHOO_ACCESS_TOKEN_JSON=YAHOO_ACCESS_TOKEN_JSON:latest"

# Optional secrets - comment out or add as needed (comma-separated, no spaces before =)
# SECRETS="${SECRETS},GOOGLE_CLOUD_KEY=GOOGLE_CLOUD_KEY:latest"
# SECRETS="${SECRETS},HASHTAG_BASKETBALL_USERNAME=HASHTAG_BASKETBALL_USERNAME:latest"
# SECRETS="${SECRETS},HASHTAG_BASKETBALL_PASSWORD=HASHTAG_BASKETBALL_PASSWORD:latest"
# SECRETS="${SECRETS},LANGSMITH_API_KEY=LANGSMITH_API_KEY:latest"
# SECRETS="${SECRETS},FIRECRAWL_API_KEY=FIRECRAWL_API_KEY:latest"

# Non-sensitive env vars (production values)
# YAHOO_REDIRECT_URI must match your Yahoo app's allowed redirect URIs
# RUNTIME_ENVIRONMENT=docker tells yfpy not to open a browser for OAuth (use tokens only)
ENV_VARS="YAHOO_REDIRECT_URI=https://www.clutchai.app,YAHOO_LEAGUE_ID=58930,RUNTIME_ENVIRONMENT=docker,GOOGLE_CLOUD_PROJECT=clutchai-480619,CLOUDSQL_REGION=us-central1,CLOUDSQL_INSTANCE=clutchai-db,CLOUDSQL_DATABASE=clutchai_db,CLOUDSQL_VECTOR_TABLE=vectorstore,CLOUDSQL_APP_TABLE=app,CLOUDSQL_USER=clutchai_user"

echo "Deploying ClutchAI to Cloud Run..."
gcloud run deploy clutchai \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-secrets="$SECRETS" \
  --set-env-vars="$ENV_VARS"

echo ""
echo "Deploy complete. Service URL:"
gcloud run services describe clutchai --region us-central1 --format='value(status.url)'
