#!/usr/bin/env python3
"""
Build YAHOO_ACCESS_TOKEN_JSON from .env for Cloud Run.

Run OAuth once locally with YAHOO_REDIRECT_URI=https://www.clutchai.app so yfpy
writes token vars to .env. Then run this script to output the JSON to store in
Secret Manager as YAHOO_ACCESS_TOKEN_JSON.

Usage:
  python scripts/pipelines/build_yahoo_token_json.py [.env path]
  # Copy the single line of JSON output and create the secret:
  echo -n '<paste JSON>' | gcloud secrets create YAHOO_ACCESS_TOKEN_JSON --data-file=- --replication-policy=automatic
  # Or add to .env as YAHOO_ACCESS_TOKEN_JSON='...' and run update_secrets.sh (value must be one line)
"""

import json
import os
import sys
from pathlib import Path

# Required keys for yfpy (see yfpy/query.py yahoo_access_token_required_fields)
REQUIRED_KEYS = {
    "access_token",
    "consumer_key",
    "consumer_secret",
    "guid",
    "refresh_token",
    "token_time",
    "token_type",
}


def load_dotenv_simple(path: Path) -> dict:
    out = {}
    if not path.is_file():
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            out[key] = value
    return out


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    env_path = Path(sys.argv[1]) if len(sys.argv) > 1 else project_root / ".env"
    env = load_dotenv_simple(env_path)

    # Map .env names to token dict keys (yfpy writes yahoo_* and also has CONSUMER_KEY/SECRET)
    mapping = {
        "YAHOO_ACCESS_TOKEN": "access_token",
        "YAHOO_CONSUMER_KEY": "consumer_key",
        "YAHOO_CLIENT_ID": "consumer_key",  # alternative
        "YAHOO_CONSUMER_SECRET": "consumer_secret",
        "YAHOO_CLIENT_SECRET": "consumer_secret",  # alternative
        "YAHOO_GUID": "guid",
        "YAHOO_REFRESH_TOKEN": "refresh_token",
        "YAHOO_TOKEN_TIME": "token_time",
        "YAHOO_TOKEN_TYPE": "token_type",
    }
    token = {}
    for env_key, dict_key in mapping.items():
        if env_key in env and env[env_key]:
            val = env[env_key]
            if dict_key == "token_time":
                try:
                    val = float(val)
                except ValueError:
                    val = float(0)
            token[dict_key] = val
    # Prefer CONSUMER_ over CLIENT_ if both present
    if "YAHOO_CONSUMER_KEY" in env and env["YAHOO_CONSUMER_KEY"]:
        token["consumer_key"] = env["YAHOO_CONSUMER_KEY"]
    if "YAHOO_CONSUMER_SECRET" in env and env["YAHOO_CONSUMER_SECRET"]:
        token["consumer_secret"] = env["YAHOO_CONSUMER_SECRET"]

    missing = REQUIRED_KEYS - set(token.keys())
    if missing:
        print(
            "Missing required keys in .env: " + ", ".join(sorted(missing)),
            file=sys.stderr,
        )
        print(
            "Run the app locally with YAHOO_REDIRECT_URI=https://www.clutchai.app, "
            "complete Yahoo OAuth so yfpy writes token vars to .env, then run this again.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Single-line JSON for use in env / Secret Manager
    print(json.dumps(token, separators=(",", ":")))


if __name__ == "__main__":
    main()
