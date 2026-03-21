#!/bin/bash
#
# Upload .env variables to Google Cloud Secrets Manager.
# Creates new secrets or adds new versions for existing ones.
#
# Usage:
#   ./scripts/update_secrets.sh [env_file]
#
# Examples:
#   ./scripts/update_secrets.sh              # Uses .env in project root
#   ./scripts/update_secrets.sh .env.prod    # Uses custom env file
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - Secret Manager API enabled on your project
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="${1:-$PROJECT_ROOT/.env}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Error: Env file not found: $ENV_FILE"
  exit 1
fi

# Load project from .env if GOOGLE_CLOUD_PROJECT is set
if [[ -f "$ENV_FILE" ]]; then
  PROJECT=$(grep -E "^GOOGLE_CLOUD_PROJECT=" "$ENV_FILE" | cut -d'=' -f2- | tr -d '"' | tr -d "'" | xargs)
fi

PROJECT="${PROJECT:-clutchai-480619}"

echo "Using project: $PROJECT"
echo "Using env file: $ENV_FILE"
echo ""

# Set project
gcloud config set project "$PROJECT"

# Enable Secret Manager API (idempotent)
gcloud services enable secretmanager.googleapis.com --quiet 2>/dev/null || true

# Create secrets from .env
while IFS= read -r line; do
  # Skip empty lines and comments
  [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue

  # Split on first = only (values may contain =)
  key="${line%%=*}"
  value="${line#*=}"

  # Trim whitespace from key
  key=$(echo "$key" | xargs)

  # Remove surrounding quotes from value
  value=$(echo "$value" | sed -e "s/^['\"]//" -e "s/['\"]$//")

  # Skip if key or value is empty
  [[ -z "$key" || -z "$value" ]] && continue

  # Create secret or add new version if it exists
  if echo -n "$value" | gcloud secrets create "$key" \
    --data-file=- \
    --replication-policy=automatic \
    2>/dev/null; then
    echo "Created: $key"
  elif echo -n "$value" | gcloud secrets versions add "$key" --data-file=- 2>/dev/null; then
    echo "Updated: $key"
  else
    echo "Failed: $key"
  fi
done < "$ENV_FILE"

echo ""
echo "Done."
