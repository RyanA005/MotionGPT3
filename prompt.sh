#!/usr/bin/env bash
set -e
curl -sf "${SERVER:-http://127.0.0.1:8888}/generate/artifacts" \
  -H "Content-Type: application/json" \
  -d "$(python -c "import json,sys; print(json.dumps({'prompt':sys.argv[1],'task':'t2m','motion_length':sys.argv[2]}))" "$1" "$2")"
echo
