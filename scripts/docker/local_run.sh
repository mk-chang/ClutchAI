#!/bin/bash
# Run ClutchAI locally via Docker (connected to Google Cloud SQL)
#
# Requires: docker image built (run ./scripts/docker/local_build.sh first)
# Requires: gcloud auth application-default login
#
# To stop: Ctrl+C (or: docker stop clutchai-local)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

ENV_FILE="$PROJECT_ROOT/.env"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Error: .env file not found at $ENV_FILE"
  echo "Copy env.example to .env and add your credentials."
  exit 1
fi

ADC_FILE="${HOME}/.config/gcloud/application_default_credentials.json"
if [[ ! -f "$ADC_FILE" ]]; then
  echo "Error: GCP credentials not found at $ADC_FILE"
  echo "Run: gcloud auth application-default login"
  exit 1
fi

echo "Running ClutchAI container..."
docker rm -f clutchai-local 2>/dev/null || true
docker run --rm --name clutchai-local -p 8080:8080 \
  --env-file "$ENV_FILE" \
  -v "$ENV_FILE:/app/.env" \
  -v "$ADC_FILE:/app/gcloud-adc.json:ro" \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/gcloud-adc.json \
  clutchai
