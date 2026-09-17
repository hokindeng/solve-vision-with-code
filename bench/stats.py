#!/usr/bin/env python3
"""Per-lane run statistics for the paper: attempts, produced, timeouts, tool calls, tokens, wall time,
joined with the official scores.

    python3 bench/stats.py <runs_root> bench/stats.json

<runs_root>/<lane>/<split>/<task>/<idx>/{summary.json, app/events.jsonl, app/exit_code}
Event formats: codex (turn.completed.usage, item.* tool events), claude (stream-json: assistant tool_use
blocks, result.usage), gemini (result.stats.tool_calls / tokens), opencode (step_finish.part.tokens,
tool_use events).
"""
import json, statistics as st, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def agent_of(lane: str) -> str:
    return "codex" if lane.startswith("codex") else "claude" if lane.startswith("claude") else "gemini" if lane.startswith("gemini") else "opencode"


def parse_events(path: Path, agent: str) -> dict:
    tools = 0; inp = out = 0; found = False
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except FileNotFoundError:
        return dict(tools=0, input_tokens=0, output_tokens=0, has_events=False)
    for line in lines:
        try:
            e = json.loads(line)
        except Exception:
            continue
        found = True
        t = e.get("type", "")
        if agent == "codex":
            if t == "item.completed" and e.get("item", {}).get("type") in ("command_execution", "file_change", "mcp_tool_call", "web_search"):
                tools += 1
            if t == "turn.completed":
                u = e.get("usage", {}); inp += u.get("input_tokens", 0); out += u.get("output_tokens", 0)
        elif agent == "claude":
            if t == "assistant":
                for b in e.get("message", {}).get("content", []) or []:
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        tools += 1
            if t == "result":
                u = e.get("usage", {}); inp += u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0) + u.get("cache_read_input_tokens", 0); out += u.get("output_tokens", 0)
        elif agent == "gemini":
            if t == "tool_use":
                tools += 1
            if t == "result":
                s = e.get("stats", {}); inp += s.get("input_tokens", 0); out += s.get("output_tokens", 0)
        else:  # opencode
            if t == "tool_use":
                tools += 1
            if t == "step_finish":
                tk = e.get("part", {}).get("tokens", {}); inp += tk.get("input", 0) + tk.get("cache", {}).get("read", 0); out += tk.get("output", 0) + tk.get("reasoning", 0)
    return dict(tools=tools, input_tokens=inp, output_tokens=out, has_events=found)


def lane_stats(lane_dir: Path, scores: dict) -> dict:
    agent = agent_of(lane_dir.name)
    rows = []
    for s in lane_dir.glob("*/*/*/summary.json"):
        d = json.loads(s.read_text())
        if d.get("skipped"):
            continue
        ev = parse_events(s.parent / "app" / "events.jsonl", agent)
        key = (s.parent.parent.name, s.parent.name)
        rows.append(dict(task=key[0], idx=key[1], seconds=d.get("seconds", 0), exit=d.get("agent_exit"), video=bool(d.get("video")),
                         timeout=d.get("agent_exit") == -9 or d.get("seconds", 0) >= 1799, score=scores.get(key), **ev))
    if not rows:
        return {}
    n = len(rows)
    produced = [r for r in rows if r["video"]]
    used_tools = [r for r in rows if r["tools"] > 0]
    return dict(
        lane=lane_dir.name, agent=agent, attempts=n, produced=len(produced), produced_rate=len(produced) / n,
        timeouts=sum(r["timeout"] for r in rows), tool_use_rate=len(used_tools) / n,
        tools_per_attempt=st.mean(r["tools"] for r in rows), tools_per_produced=st.mean(r["tools"] for r in produced) if produced else 0,
        seconds_median=st.median(r["seconds"] for r in rows), seconds_mean=st.mean(r["seconds"] for r in rows),
        seconds_median_produced=st.median(r["seconds"] for r in produced) if produced else 0,
        input_tokens_mean=st.mean(r["input_tokens"] for r in rows), output_tokens_mean=st.mean(r["output_tokens"] for r in rows),
        score_over_500=sum((r["score"] or 0.0) for r in rows) / 500, score_over_produced=st.mean(r["score"] for r in produced if r["score"] is not None) if produced else 0,
        no_video_no_tools=sum(1 for r in rows if not r["video"] and r["tools"] == 0),
        no_video_with_tools=sum(1 for r in rows if not r["video"] and r["tools"] > 0),
    )


def main():
    runs_root, out = Path(sys.argv[1]), Path(sys.argv[2])
    lanes = {}
    for lane_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        res = ROOT / "bench" / "results" / lane_dir.name / f"{lane_dir.name}_vbvr_results.json"
        scores = {}
        if res.exists():
            for x in json.load(open(res))["samples"]:
                scores[(x["task_name"], x["video_file"].replace(".mp4", ""))] = x["score"]
        s = lane_stats(lane_dir, scores)
        if s:
            lanes[lane_dir.name] = s
            print(f"{lane_dir.name:34s} n={s['attempts']:3d} produced={s['produced']:3d} timeouts={s['timeouts']:3d} tool_use={s['tool_use_rate']:.2f} tools/att={s['tools_per_attempt']:5.1f} med_s={s['seconds_median']:5.0f} in_tok={s['input_tokens_mean']:8.0f} out_tok={s['output_tokens_mean']:6.0f} score={s['score_over_500']:.3f}")
    out.write_text(json.dumps(lanes, indent=1))


if __name__ == "__main__":
    main()
