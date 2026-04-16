#!/usr/bin/env bash
set -e

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 \"prompt\" <motion_length_frames>"
  exit 1
fi

PROMPT="$1"
MOTION_LENGTH="$2"

curl -sf "${SERVER:-http://127.0.0.1:8888}/generate/artifacts" \
  -H "Content-Type: application/json" \
  -d "$(python -c "import json,sys; print(json.dumps({'prompt':sys.argv[1],'task':'t2m','motion_length':int(sys.argv[2])}))" "$PROMPT" "$MOTION_LENGTH")"
echo
