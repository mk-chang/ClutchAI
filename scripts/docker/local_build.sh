#!/bin/bash
# Build the ClutchAI Docker image.
# Run this when you've made code changes.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "Building ClutchAI image..."
docker build -t clutchai .
