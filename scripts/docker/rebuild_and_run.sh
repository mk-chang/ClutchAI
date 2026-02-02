#!/bin/bash
# Rebuild and run ClutchAI locally via Docker.
# Use this when you've made code changes and want a fresh run.
#
# For quick runs without rebuild, use: ./scripts/docker/local_run.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/local_build.sh" && "$SCRIPT_DIR/local_run.sh"