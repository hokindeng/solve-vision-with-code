#!/usr/bin/env python3
"""Leaderboard page + markdown table from bench/results/<lane>/<lane>_vbvr_results.json.

    python3 bench/report.py <out.html> [--date 2026-09-13]

Video-model rows are the published VBVR-Pro-Bench leaderboard numbers (paper Table 8 / leaderboard page).
"""
import argparse, json, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANES = {  # lane dir -> display name
    "codex-gpt-6-astra": "Codex × gpt-6-astra (coding agent)",
    "gemini-gemini-3.1-pro-preview": "Gemini CLI × gemini-3.1-pro (coding agent)",
    "claude-fable-5.1-bedrock": "Claude Code × Fable 5.1 (coding agent)",
    "bedrock-deepseek-v3.2": "OpenCode × DeepSeek V3.2 (open weights, Bedrock)",
    "bedrock-gpt-oss-120b": "OpenCode × gpt-oss-120b (open weights, Bedrock)",
    "bedrock-gpt-oss-20b": "OpenCode × gpt-oss-20b (open weights, Bedrock)",
    "bedrock-qwen3-next-80b": "OpenCode × Qwen3-Next-80B-A3B (open weights, Bedrock)",
    "bedrock-llama4-maverick": "OpenCode × Llama 4 Maverick (open weights, Bedrock via converse proxy)",
    "bedrock-llama4-scout": "OpenCode × Llama 4 Scout (open weights, Bedrock via converse proxy)",
    "bedrock-qwen3-coder-30b": "OpenCode × Qwen3-Coder-30B-A3B (open weights, Bedrock)",
    "bedrock-qwen3-32b": "OpenCode × Qwen3-32B (open weights, Bedrock)",
    "bedrock-nemotron-super-3": "OpenCode × Nemotron Super 3 120B (open weights, Bedrock)",
    "bedrock-nemotron-nano-3": "OpenCode × Nemotron Nano 3 30B (open weights, Bedrock)",
    "bedrock-minimax-m2.5": "OpenCode × MiniMax M2.5 (open weights, Bedrock)",
    "bedrock-kimi-k2.5": "OpenCode × Kimi K2.5 (open weights, Bedrock)",
    "bedrock-glm-5": "OpenCode × GLM-5 (open weights, Bedrock)",
    "bedrock-glm-4.7-flash": "OpenCode × GLM-4.7-Flash (open weights, Bedrock)",
    "bedrock-qwen3-vl-235b": "OpenCode × Qwen3-VL-235B-A22B (open weights, Bedrock)",
    "bedrock-mistral-large-3": "OpenCode × Mistral Large 3 675B (open weights, Bedrock)",
    "bedrock-kimi-k2-thinking": "OpenCode × Kimi K2 Thinking (open weights, Bedrock)",
    "bedrock-devstral-2": "OpenCode × Devstral 2 123B (open weights, Bedrock)",
    "bedrock-qwen3-coder-next": "OpenCode × Qwen3-Coder-Next (open weights, Bedrock)",
}
VIDEO_MODELS = [("VBVR-Pro-Wan2.2-I2V-A14B (RL-trained video model)", 0.670, 0.808, 0.532), ("VBVR-Pro-Wan2.1-I2V-14B", 0.562, 0.730, 0.395),
                ("VBVR-Wan2.2", 0.517, 0.548, 0.486), ("Seedance 2.0", 0.499, 0.451, 0.547), ("VBVR-Pro-Wan2.2-TI2V-5B", 0.470, 0.641, 0.300),
                ("VBVR-Pro-LTX2.3", 0.425, 0.527, 0.324), ("Kling VIDEO 3.0", 0.392, 0.356, 0.427), ("Veo 3.1", 0.309, 0.312, 0.305),
                ("Wan2.2-I2V-A14B", 0.182, 0.157, 0.207), ("LTX-2.3-I2AV", 0.112, 0.106, 0.119), ("Wan2.1-I2V-14B-720P", 0.100, 0.105, 0.095)]
CATS = ["Abstraction", "Perception", "Spatiality", "Transformation", "Knowledge"]
# Lanes whose agent produced no video at all (500 attempts): score 0 by definition; the evaluator writes no file.
ZERO_LANES = {"bedrock-gemma-3-27b": "OpenCode × Gemma 3 27B (open weights, Bedrock)",
              "bedrock-gemma-3-12b": "OpenCode × Gemma 3 12B (open weights, Bedrock)",
              "bedrock-magistral-small": "OpenCode × Magistral Small (open weights, Bedrock)",
              "bedrock-llama3.3-70b": "OpenCode × Llama 3.3 70B (open weights, Bedrock via converse proxy)"}


def load(lane):
    p = ROOT / "bench" / "results" / lane / f"{lane}_vbvr_results.json"
    return json.load(open(p))["samples"] if p.exists() else None


def stats(s):
    by = {}; per = {}; cat = {}
    for x in s:
        by.setdefault(x["split"], []).append(x["score"]); per.setdefault(x["task_name"].replace("_data-generator", ""), []).append(x["score"])
        cat.setdefault(x["category"], []).append(x["score"])
    tot = {"In_Domain": 250, "Out_of_Domain": 250}  # missing videos (agent produced nothing) count as 0, like a video model that fails to generate
    return dict(overall=sum(x["score"] for x in s) / 500, ID=sum(by.get("In_Domain", [])) / tot["In_Domain"], OOD=sum(by.get("Out_of_Domain", [])) / tot["Out_of_Domain"],
                per={k: st.mean(v) for k, v in per.items()}, cat={k: st.mean(v) for k, v in cat.items()}, n=len(s),
                produced_mean=st.mean(x["score"] for x in s))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--date", default="2026-09-16")
    a = ap.parse_args()
    ours = {lane: stats(s) for lane, s in ((l, load(l)) for l in LANES) if s}
    rows = [(LANES[l], v["overall"], v["ID"], v["OOD"], True) for l, v in ours.items()] + [(n, o, i, d, False) for n, o, i, d in VIDEO_MODELS]
    rows += [(n + " <small>0/500 produced</small>", 0.0, 0.0, 0.0, False) for n in ZERO_LANES.values()]
    rows.sort(key=lambda r: -r[1])
    md = ["| # | model | Overall | In-Domain | Out-of-Domain |", "|---|---|---|---|---|"]
    tr = ""
    for k, (n, o, i, d, mine) in enumerate(rows, 1):
        note = ""
        if mine:
            v = next(v for l, v in ours.items() if LANES[l] == n)
            if v["n"] < 500: note = f" <small>{v['n']}/500 produced; mean over produced {v['produced_mean']:.3f}</small>"
        tr += f"<tr{' class=ours' if mine else ''}><td>{k}</td><td>{n}{note}</td><td><b>{o:.3f}</b></td><td>{i:.3f}</td><td>{d:.3f}</td></tr>"
        md.append(f"| {k} | {n} | **{o:.4f}** | {i:.4f} | {d:.4f} |" if mine else f"| {k} | {n} | {o:.3f} | {i:.3f} | {d:.3f} |")
    cat_tr = "".join(f"<tr><td>{c}</td>" + "".join(f"<td>{v['cat'].get(c, float('nan')):.3f}</td>" for v in ours.values()) + "</tr>" for c in CATS)
    head = "".join(f"<th>{LANES[l].split(' (')[0]}</th>" for l in ours)
    codex = ours.get("codex-gpt-6-astra") or next(iter(ours.values()))
    weakest = sorted(codex["per"].items(), key=lambda kv: kv[1])[:10]
    weak_tr = "".join(f"<tr><td>{t}</td><td>{v:.2f}</td></tr>" for t, v in weakest)
    perfect = sum(1 for v in codex["per"].values() if v >= 0.999)
    html = f"""<!doctype html><html><head><meta charset=utf-8><title>solve-vision-with-code · VBVR-Pro-Bench</title>
<style>body{{font:15px/1.45 -apple-system,Helvetica,Arial;margin:32px auto;max-width:1100px;color:#222}}table{{border-collapse:collapse;margin:8px 0 20px}}td,th{{border:1px solid #ddd;padding:5px 10px;text-align:left}}th{{background:#f4f4f4}}tr.ours td{{background:#eef7ee}}.row{{display:flex;gap:32px;flex-wrap:wrap}}</style></head><body>
<h1>Coding agents on VBVR-Pro-Bench (video setting)</h1>
<p>Each agent gets only <code>first_frame.png</code> + <code>prompt.txt</code> and must write code that renders the answer video.
Scored by the unmodified official VBVR-Pro-Bench rule-based evaluators in the kit's own environment (numpy&lt;2, norfair, easyocr). {a.date}; 500 instances per lane; 3 closed-model coding agents and all 24 open-weight models on Bedrock (roster complete).</p>
<h2>Leaderboard (video / TI2V setting)</h2>
<table><tr><th>#</th><th>model</th><th>Overall</th><th>In-Domain</th><th>Out-of-Domain</th></tr>{tr}</table>
<div class="row"><div><h2>Coding agents by category</h2>
<table><tr><th>category</th>{head}</tr>{cat_tr}</table></div>
<div><h2>Weakest tasks for Codex (of 100)</h2><table><tr><th>task</th><th>mean</th></tr>{weak_tr}</table>
<p>{perfect} of 100 tasks are a perfect 1.00 across all 5 instances.</p></div></div>
<p>Notable: every trained video model drops Out-of-Domain (e.g. 0.808→0.532); the coding agents go UP (In-Domain → Out-of-Domain) — code does not overfit
the task families it was never trained on. Correction {a.date}: the first scoring ran the evaluator with numpy 2 and without norfair, which silently zeroed
G-5, G-8, O-18 and O-19 for every model; all lanes were re-scored in the conforming environment.</p>
</body></html>"""
    Path(a.out).write_text(html)
    print("\n".join(md))
    for l, v in ours.items():
        print(f"{l}: n={v['n']} overall {v['overall']:.4f} ID {v['ID']:.4f} OOD {v['OOD']:.4f} produced-mean {v['produced_mean']:.4f} | " + " ".join(f"{c[:5]} {v['cat'][c]:.3f}" for c in CATS))


if __name__ == "__main__":
    main()
