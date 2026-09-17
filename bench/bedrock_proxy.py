#!/usr/bin/env python3
"""OpenAI-compatible chat endpoint in front of Amazon Bedrock's non-streaming Converse API.

Why: Bedrock's own OpenAI-compatible endpoint returns 404 for some models (deepseek-r1, llama), and its
streaming Converse API rejects tool use for them ("This model doesn't support tool use in streaming mode").
OpenCode always streams, so it cannot drive those models directly. This proxy accepts the OpenAI
chat-completions request OpenCode sends (messages + tools, stream=true), calls `converse` (non-streaming),
and replays the answer as a short SSE stream. Auth: the same Bedrock bearer token, read by boto3 from
AWS_BEARER_TOKEN_BEDROCK; the client-side Authorization header is ignored.

    AWS_BEARER_TOKEN_BEDROCK=... python3 bench/bedrock_proxy.py --port 8765 [--region us-east-1]
    # then: SVC_BASE_URL=http://172.17.0.1:8765/v1 SVC_API_KEY=x  (host bridge address as seen from containers)

Only what OpenCode needs is translated: system/user/assistant/tool messages, text and image_url parts (PNG/JPEG
data URLs), function tools, tool_choice auto, max_tokens, temperature. Reasoning-model output ("reasoningContent")
is dropped from the content; tool calls map 1:1.
"""
import argparse, base64, json, os, re, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import boto3
from botocore.config import Config

REGION = os.environ.get("AWS_REGION", "us-east-1")
_client = None
_lock = threading.Lock()


def client():
    global _client
    with _lock:
        if _client is None:
            _client = boto3.client("bedrock-runtime", region_name=REGION, config=Config(read_timeout=600, retries={"max_attempts": 3}))
        return _client


def _image_block(url: str):
    m = re.match(r"data:image/(png|jpeg|jpg|gif|webp);base64,(.*)", url, re.S)
    if not m:
        return {"text": f"[image {url[:60]}]"}
    fmt = "jpeg" if m.group(1) == "jpg" else m.group(1)
    return {"image": {"format": fmt, "source": {"bytes": base64.b64decode(m.group(2))}}}


def to_converse(messages):
    """OpenAI messages -> (system blocks, converse messages). Consecutive same-role turns are merged."""
    system, out = [], []

    def push(role, blocks):
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": blocks})

    for m in messages:
        role, content = m.get("role"), m.get("content")
        if role == "system":
            system.append({"text": content if isinstance(content, str) else " ".join(p.get("text", "") for p in content)})
        elif role in ("user", "assistant"):
            blocks = []
            if isinstance(content, str):
                if content:
                    blocks.append({"text": content})
            elif content:
                for p in content:
                    if p.get("type") == "text" and p.get("text"):
                        blocks.append({"text": p["text"]})
                    elif p.get("type") == "image_url":
                        blocks.append(_image_block(p["image_url"]["url"]))
            for tc in m.get("tool_calls") or []:
                fn = tc["function"]
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": fn.get("arguments")}
                blocks.append({"toolUse": {"toolUseId": tc["id"], "name": fn["name"], "input": args}})
            if blocks:
                push(role, blocks)
        elif role == "tool":
            text = content if isinstance(content, str) else json.dumps(content)
            push("user", [{"toolResult": {"toolUseId": m["tool_call_id"], "content": [{"text": text or "(empty)"}]}}])
    if out and out[0]["role"] != "user":
        out.insert(0, {"role": "user", "content": [{"text": "(start)"}]})
    return system, out


def to_tool_config(tools, tool_choice):
    if not tools:
        return None
    specs = []
    for t in tools:
        fn = t["function"]
        schema = fn.get("parameters") or {"type": "object", "properties": {}}
        specs.append({"toolSpec": {"name": fn["name"], "description": (fn.get("description") or fn["name"])[:1000], "inputSchema": {"json": schema}}})
    cfg = {"tools": specs}
    if tool_choice == "required":
        cfg["toolChoice"] = {"any": {}}
    elif isinstance(tool_choice, dict):
        cfg["toolChoice"] = {"tool": {"name": tool_choice["function"]["name"]}}
    return cfg


def complete(req):
    system, msgs = to_converse(req["messages"])
    kw = {"modelId": req["model"], "messages": msgs}
    if system:
        kw["system"] = system
    inf = {}
    if req.get("max_tokens") or req.get("max_completion_tokens"):
        inf["maxTokens"] = int(min(req.get("max_tokens") or req.get("max_completion_tokens"), int(os.environ.get("SVC_PROXY_MAX_TOKENS", "8192"))))
    if req.get("temperature") is not None:
        inf["temperature"] = float(req["temperature"])
    if inf:
        kw["inferenceConfig"] = inf
    tc = to_tool_config(req.get("tools"), req.get("tool_choice", "auto"))
    if tc:
        kw["toolConfig"] = tc
    r = client().converse(**kw)
    text, calls = [], []
    for b in r["output"]["message"]["content"]:
        if "text" in b:
            text.append(b["text"])
        elif "toolUse" in b:
            tu = b["toolUse"]
            calls.append({"id": tu["toolUseId"], "type": "function", "function": {"name": tu["name"], "arguments": json.dumps(tu["input"])}})
    finish = {"tool_use": "tool_calls", "max_tokens": "length"}.get(r.get("stopReason"), "stop")
    u = r.get("usage", {})
    return "".join(text), calls, finish, {"prompt_tokens": u.get("inputTokens", 0), "completion_tokens": u.get("outputTokens", 0), "total_tokens": u.get("totalTokens", 0)}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            return self._json(200, {"object": "list", "data": []})
        self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            return self._json(404, {"error": {"message": "not found"}})
        n = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(n) or b"{}")
        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"; now = int(time.time())
        try:
            text, calls, finish, usage = complete(req)
        except Exception as e:  # surface Bedrock's message in OpenAI's error shape
            msg = getattr(e, "response", {}).get("Error", {}).get("Message", str(e)) if hasattr(e, "response") else str(e)
            code = getattr(e, "response", {}).get("ResponseMetadata", {}).get("HTTPStatusCode", 500) if hasattr(e, "response") else 500
            return self._json(code if isinstance(code, int) else 500, {"error": {"message": msg, "type": "bedrock_error"}})
        if not req.get("stream"):
            msg = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            return self._json(200, {"id": cid, "object": "chat.completion", "created": now, "model": req["model"], "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage})
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-cache"); self.send_header("Connection", "keep-alive"); self.send_header("Transfer-Encoding", "chunked"); self.end_headers()

        def chunk(delta, fin=None):
            payload = {"id": cid, "object": "chat.completion.chunk", "created": now, "model": req["model"], "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]}
            data = f"data: {json.dumps(payload)}\n\n".encode()
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")

        chunk({"role": "assistant", "content": ""})
        if text:
            chunk({"content": text})
        for i, c in enumerate(calls):
            chunk({"tool_calls": [{"index": i, "id": c["id"], "type": "function", "function": {"name": c["function"]["name"], "arguments": ""}}]})
            chunk({"tool_calls": [{"index": i, "function": {"arguments": c["function"]["arguments"]}}]})
        chunk({}, finish)
        tail = f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': now, 'model': req['model'], 'choices': [], 'usage': usage})}\n\ndata: [DONE]\n\n".encode()
        self.wfile.write(f"{len(tail):x}\r\n".encode() + tail + b"\r\n0\r\n\r\n"); self.wfile.flush()


def main():
    global REGION
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8765); ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--region", default=REGION)
    a = ap.parse_args()
    REGION = a.region
    assert os.environ.get("AWS_BEARER_TOKEN_BEDROCK"), "set AWS_BEARER_TOKEN_BEDROCK"
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
