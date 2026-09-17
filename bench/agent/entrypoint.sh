#!/usr/bin/env bash
# Runs one agent CLI on one task. /app holds first_frame.png, prompt.txt and instruction.md; the agent
# works there and leaves output/, solve.py, events.jsonl (its raw stream) and agent.stderr.
# SVC_AGENT picks the shell: codex (default) | claude | gemini | opencode. Plain official CLI calls.
set -uo pipefail
cd /app
: "${SVC_MODEL:?set SVC_MODEL}"
AGENT="${SVC_AGENT:-codex}"
export HOME="${HOME:-/tmp/home}"; mkdir -p "$HOME"
mkdir -p output
date -u +%Y-%m-%dT%H:%M:%SZ > started_at
case "$AGENT" in
  codex)
    : "${CODEX_API_KEY:=${OPENAI_API_KEY:-}}"; export CODEX_API_KEY
    args=(codex exec --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check -C /app
          -m "$SVC_MODEL" -i first_frame.png --json -o final.md)
    [ -n "${SVC_REASONING:-}" ] && args+=(-c "model_reasoning_effort=\"$SVC_REASONING\"")
    "${args[@]}" - < instruction.md > events.jsonl 2> agent.stderr
    rc=$?
    ;;
  claude)
    export DISABLE_TELEMETRY=1 DISABLE_AUTOUPDATER=1
    if [ "${CLAUDE_CODE_USE_BEDROCK:-}" = "1" ]; then  # Bedrock route: AWS_* creds + region arrive in the env
      export ANTHROPIC_MODEL="$SVC_MODEL" ANTHROPIC_DEFAULT_OPUS_MODEL=us.anthropic.claude-opus-5 \
             ANTHROPIC_DEFAULT_SONNET_MODEL=us.anthropic.claude-sonnet-5 ANTHROPIC_DEFAULT_HAIKU_MODEL=us.anthropic.claude-sonnet-5 \
             ANTHROPIC_SMALL_FAST_MODEL=us.anthropic.claude-sonnet-5
    fi
    claude --print "$(cat instruction.md)" --output-format stream-json --verbose \
           --permission-mode bypassPermissions --model "$SVC_MODEL" > events.jsonl 2> agent.stderr
    rc=$?
    ;;
  gemini)
    export GEMINI_CLI_DISABLE_AUTOUPDATE=1 GEMINI_CLI_TRUST_WORKSPACE=true
    mkdir -p "$HOME/.gemini"
    printf '{"security":{"auth":{"selectedType":"gemini-api-key"}},"general":{"disableAutoUpdate":true,"disableUpdateNag":true},"privacy":{"usageStatisticsEnabled":false}}' > "$HOME/.gemini/settings.json"
    gemini --prompt "$(cat instruction.md)" --output-format stream-json --approval-mode yolo --skip-trust \
           --model "$SVC_MODEL" > events.jsonl 2> agent.stderr
    rc=$?
    ;;
  opencode)
    # SVC_PROVIDER=openai (default): any OpenAI-compatible endpoint at SVC_BASE_URL.
    # SVC_PROVIDER=bedrock: Amazon Bedrock's native (converse) API via @ai-sdk/amazon-bedrock, for models the
    # OpenAI-compatible endpoint does not serve (deepseek-r1, llama). Same bearer token (SVC_API_KEY).
    : "${SVC_API_KEY:?set SVC_API_KEY}"; SVC_PROVIDER="${SVC_PROVIDER:-openai}"
    [ "$SVC_PROVIDER" = bedrock ] || : "${SVC_BASE_URL:?set SVC_BASE_URL (OpenAI-compatible endpoint)}"
    export XDG_CONFIG_HOME="$HOME/.config" XDG_DATA_HOME="$HOME/.local/share" XDG_CACHE_HOME="$HOME/.cache"
    mkdir -p "$XDG_CONFIG_HOME/opencode" "$XDG_DATA_HOME" "$XDG_CACHE_HOME"
    [ -e "$XDG_CONFIG_HOME/opencode/node_modules" ] || cp -R /opt/svc/opencode-config/. "$XDG_CONFIG_HOME/opencode/"
    export OPENCODE_CONFIG="$HOME/opencode.json" OPENCODE_DISABLE_AUTOUPDATE=1 OPENCODE_DISABLE_MODELS_FETCH=1 OPENCODE_DISABLE_SHARE=1
    python3 - "$SVC_PROVIDER" "${SVC_BASE_URL:-}" "${SVC_BEDROCK_REGION:-us-east-1}" "$SVC_MODEL" "$OPENCODE_CONFIG" <<'PYEOF'
import json, os, sys
prov, base, region, model, path = sys.argv[1:6]
# Bedrock's native API rejects max_tokens above the model limit (llama: 8192); SVC_MAX_OUTPUT overrides.
out = int(os.environ.get("SVC_MAX_OUTPUT", 8192 if prov == "bedrock" else 16384))
models = {model: {"name": model, "tool_call": True, "attachment": True, "limit": {"context": 128000, "output": out}}}
if prov == "bedrock":
    provider = {"npm": "@ai-sdk/amazon-bedrock", "name": "svc", "options": {"region": region, "apiKey": "{env:SVC_API_KEY}"}, "models": models}
else:
    provider = {"npm": "@ai-sdk/openai-compatible", "name": "svc", "options": {"baseURL": base, "apiKey": "{env:SVC_API_KEY}"}, "models": models}
cfg = {"$schema": "https://opencode.ai/config.json", "autoupdate": False, "share": "disabled", "provider": {"svc": provider}}
open(path, "w").write(json.dumps(cfg))
PYEOF
    opencode run --pure --auto --format json --model "svc/$SVC_MODEL" "$(cat instruction.md)" > events.jsonl 2> agent.stderr
    rc=$?
    ;;
  *) echo "unknown SVC_AGENT=$AGENT" >&2; exit 64 ;;
esac
echo $rc > exit_code
date -u +%Y-%m-%dT%H:%M:%SZ > finished_at
