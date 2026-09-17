# Agent runner: the task environment (python:3.11 + ffmpeg + imaging libs + fonts) plus Codex CLI.
# One container per task; only /app (the agent workspace) is mounted. The key arrives as an env var.
FROM python:3.11-slim
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git ffmpeg libgl1 libglib2.0-0 fonts-dejavu-core fonts-liberation fonts-noto-cjk \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*
ARG CODEX_VERSION=0.154.0
ARG CLAUDE_VERSION=2.1.268
ARG GEMINI_VERSION=0.59.0
ARG OPENCODE_VERSION=1.18.30
RUN npm install -g "@openai/codex@${CODEX_VERSION}" "@anthropic-ai/claude-code@${CLAUDE_VERSION}" \
        "@google/gemini-cli@${GEMINI_VERSION}" "opencode-ai@${OPENCODE_VERSION}" && rm -rf /root/.npm
# opencode fetches its provider SDK into $XDG_CONFIG_HOME/opencode on first run; pre-install it once here
# and let the entrypoint copy it, so thousands of container runs do not each hit npm.
RUN mkdir -p /opt/svc/opencode-config && cd /opt/svc/opencode-config \
    && npm install --omit=dev --no-audit --no-fund @ai-sdk/openai-compatible @ai-sdk/amazon-bedrock "@opencode-ai/plugin@${OPENCODE_VERSION}" \
    && rm -rf /root/.npm && chmod -R a+rX /opt/svc
RUN pip install numpy pillow opencv-python-headless imageio imageio-ffmpeg scipy matplotlib
RUN useradd -m -u 1000 agent && mkdir -p /app /logs && chown -R agent:agent /app /logs
COPY entrypoint.sh /sandbox/entrypoint.sh
RUN chmod +x /sandbox/entrypoint.sh
USER agent
WORKDIR /app
ENTRYPOINT ["/sandbox/entrypoint.sh"]
