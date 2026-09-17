#!/usr/bin/env bash
# Run every open-weight Bedrock cell in bench/roster_bedrock_open.json through OpenCode, one after another,
# writing to bench/outputs/<cell>/. Needs SVC_API_KEY (Bedrock bearer token) in the environment.
#   bench/run_roster.sh [parallel] [cells...]
set -uo pipefail
cd "$(dirname "$0")/.."
PAR="${1:-12}"; shift || true
: "${SVC_API_KEY:?set SVC_API_KEY (Bedrock API key)}"
export SVC_OPENAI_URL="${SVC_BASE_URL:-https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1}"
PY="${PY:-python3}"
cells="$*"
[ -n "$cells" ] || cells=$($PY -c "import json;print(' '.join(json.load(open('bench/roster_bedrock_open.json'))['cells']))")
for cell in $cells; do
  model=$($PY -c "import json;print(json.load(open('bench/roster_bedrock_open.json'))['cells']['$cell']['model'])")
  route=$($PY -c "import json;print(json.load(open('bench/roster_bedrock_open.json'))['cells']['$cell'].get('route','bedrock'))")
  export SVC_PROVIDER=openai
  case "$route" in
    unsupported) echo "=== $cell skipped: unsupported"; continue ;;
    bedrock-converse) export SVC_PROVIDER=bedrock ;;
    bedrock-proxy) export SVC_BASE_URL="${SVC_PROXY_URL:-http://172.17.0.1:8765/v1}" ;;   # bench/bedrock_proxy.py on the docker host
    *) export SVC_BASE_URL="${SVC_OPENAI_URL:-https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1}" ;;
  esac
  echo "=== $cell ($model, $SVC_PROVIDER) $(date -u +%FT%TZ)"
  $PY bench/run_bench.py --agent opencode --model "$model" --name "$cell" --parallel "$PAR" --no-build --timeout 1800
done
echo "=== roster done $(date -u +%FT%TZ)"
