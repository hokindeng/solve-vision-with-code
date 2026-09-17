#!/usr/bin/env python3
"""Paper tables and figures from bench/results + bench/stats.json.

    python3 bench/paper_tables.py <out_dir>   # writes CSV/markdown tables + an HTML page with inline SVG figures
"""
import csv, json, statistics as st, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
from report import LANES, VIDEO_MODELS, ZERO_LANES, CATS, load, stats  # noqa: E402

CLOSED = ["codex-gpt-6-astra", "claude-fable-5.1-bedrock", "gemini-gemini-3.1-pro-preview"]
SHORT = {"codex-gpt-6-astra": "Codex/gpt-6-astra", "claude-fable-5.1-bedrock": "Claude Code/Fable 5.1", "gemini-gemini-3.1-pro-preview": "Gemini CLI/3.1 Pro"}


def short(lane):
    n = LANES.get(lane) or ZERO_LANES.get(lane, lane)
    return SHORT.get(lane, n.replace("OpenCode × ", "").split(" (")[0])


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def bar_svg(items, width=720, row_h=18, vmax=1.0, fmt="{:.3f}", colors=None):
    """Horizontal bars: items = [(label, value, kind)]."""
    h = row_h * len(items) + 10
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{h}" font-family="Helvetica,Arial" font-size="11">']
    for i, (label, v, kind) in enumerate(items):
        y = 5 + i * row_h; bw = (width - 330) * (v / vmax)
        col = (colors or {}).get(kind, "#888")
        parts.append(f'<text x="{300}" y="{y + 13}" text-anchor="end">{label}</text>')
        parts.append(f'<rect x="305" y="{y + 2}" width="{bw:.1f}" height="{row_h - 6}" fill="{col}"/>')
        parts.append(f'<text x="{310 + bw:.1f}" y="{y + 13}">{fmt.format(v)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    runstats = json.load(open(ROOT / "bench" / "stats.json"))
    ours = {l: stats(s) for l, s in ((l, load(l)) for l in LANES) if s}

    # 1. main table (all rows)
    rows = [(short(l), "coding agent" if l in CLOSED else "coding agent (open weights)", v["overall"], v["ID"], v["OOD"], v["n"]) for l, v in ours.items()]
    rows += [(n, "coding agent (open weights)", 0.0, 0.0, 0.0, 0) for n in ZERO_LANES.values()]
    rows += [(n, "video model", o, i, d, 500) for n, o, i, d in VIDEO_MODELS]
    rows.sort(key=lambda r: -r[2])
    write_csv(out / "table1_leaderboard.csv", ["model", "kind", "overall", "in_domain", "out_of_domain", "produced"], rows)
    t1 = "| # | model | kind | Overall | ID | OOD |\n|---|---|---|---|---|---|\n" + "\n".join(f"| {k} | {r[0]} | {r[1]} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:.3f} |" for k, r in enumerate(rows, 1))

    # 2. category table
    cat_rows = [[short(l)] + [v["cat"].get(c, 0.0) for c in CATS] + [v["overall"]] for l, v in sorted(ours.items(), key=lambda kv: -kv[1]["overall"])]
    write_csv(out / "table2_categories.csv", ["model"] + CATS + ["overall"], cat_rows)
    t2 = "| model | " + " | ".join(CATS) + " | overall |\n|---|" + "---|" * (len(CATS) + 1) + "\n" + "\n".join(f"| {r[0]} | " + " | ".join(f"{x:.3f}" for x in r[1:]) + " |" for r in cat_rows)

    # 3. efficiency / failure anatomy
    eff = []
    for l, s in sorted(runstats.items(), key=lambda kv: -kv[1]["score_over_500"]):
        eff.append([short(l), s["score_over_500"], s["produced"], s["attempts"], s["produced_rate"], s["tool_use_rate"], s["tools_per_attempt"], s["timeouts"],
                    s["seconds_median"], s["input_tokens_mean"], s["output_tokens_mean"], s["no_video_no_tools"], s["no_video_with_tools"]])
    write_csv(out / "table3_efficiency.csv", ["model", "score", "produced", "attempts", "produced_rate", "tool_use_rate", "tool_calls_per_attempt", "timeouts", "median_seconds", "mean_input_tokens", "mean_output_tokens", "failed_without_tools", "failed_with_tools"], eff)
    t3 = "| model | score | produced/500 | tool-use rate | tool calls/attempt | timeouts | median s | input tok | output tok |\n|---|---|---|---|---|---|---|---|---|\n" + "\n".join(f"| {r[0]} | {r[1]:.3f} | {r[2]} | {r[5]:.2f} | {r[6]:.1f} | {r[7]} | {r[8]:.0f} | {r[9]:,.0f} | {r[10]:,.0f} |" for r in eff)

    # 4. per-task × closed three
    tasks = sorted(ours[CLOSED[0]]["per"])
    split_of = {}
    for x in load(CLOSED[0]):
        split_of[x["task_name"].replace("_data-generator", "")] = x["split"]
    pt = [[t, split_of.get(t, "")] + [ours[l]["per"].get(t, 0.0) for l in CLOSED] for t in tasks]
    pt.sort(key=lambda r: st.mean(r[2:]))
    write_csv(out / "table4_per_task_closed.csv", ["task", "split"] + [short(l) for l in CLOSED], pt)
    hardest = pt[:15]
    t4 = "| task | split | " + " | ".join(short(l) for l in CLOSED) + " |\n|---|---|---|---|---|\n" + "\n".join(f"| {r[0]} | {r[1]} | " + " | ".join(f"{x:.2f}" for x in r[2:]) + " |" for r in hardest)

    # figures
    colors = {"closed": "#2b6cb0", "open": "#48bb78", "video": "#c05621"}
    f1 = bar_svg([(r[0], r[2], "closed" if r[1] == "coding agent" else "open" if "open" in r[1] else "video") for r in rows], colors=colors)
    ood = [(short(l) if l in ours else l, v["OOD"] - v["ID"], "closed" if l in CLOSED else "open") for l, v in ours.items()] + [(n, d - i, "video") for n, o, i, d in VIDEO_MODELS]
    ood.sort(key=lambda r: -r[1])
    f2 = bar_svg([(a, b + 0.5, c) for a, b, c in ood], vmax=1.0, fmt="{:+.3f}", colors=colors).replace("{:+.3f}", "")
    # rebuild f2 properly: show delta with sign, bars offset from 0.5
    f2 = bar_svg([(a, b, c) for a, b, c in ood], vmax=max(abs(x[1]) for x in ood) * 2.2, fmt="{:+.3f}", colors=colors)
    # heatmap for closed three
    cell = 7; hm = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{cell * len(tasks) + 160}" height="{3 * 16 + 40}" font-family="Helvetica,Arial" font-size="10">']
    order = sorted(tasks, key=lambda t: (split_of.get(t, ""), t))
    for j, l in enumerate(CLOSED):
        hm.append(f'<text x="150" y="{j * 16 + 22}" text-anchor="end">{short(l)}</text>')
        for i, t in enumerate(order):
            v = ours[l]["per"].get(t, 0.0); g = int(255 * (1 - v))
            hm.append(f'<rect x="{155 + i * cell}" y="{j * 16 + 10}" width="{cell - 1}" height="14" fill="rgb({g},{255 - int(100 * (1 - v))},{g})"><title>{t}: {v:.2f}</title></rect>')
    n_id = sum(1 for t in order if split_of.get(t) == "In_Domain")
    hm.append(f'<text x="155" y="{3 * 16 + 30}">In-Domain (50 tasks)</text><text x="{155 + n_id * cell}" y="{3 * 16 + 30}">Out-of-Domain (50 tasks)</text></svg>')
    f3 = "\n".join(hm)
    # tool calls vs score scatter (open + closed)
    pts = [(s["tools_per_attempt"], s["score_over_500"], short(l), "closed" if l in CLOSED else "open") for l, s in runstats.items() if s["produced"] > 0]
    W, H = 720, 360; xmax = max(p[0] for p in pts) * 1.1
    sc = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Helvetica,Arial" font-size="10">', f'<line x1="50" y1="{H - 30}" x2="{W - 10}" y2="{H - 30}" stroke="#999"/><line x1="50" y1="10" x2="50" y2="{H - 30}" stroke="#999"/>',
          f'<text x="{W // 2}" y="{H - 8}">mean tool calls per instance</text><text x="12" y="{H // 2}" transform="rotate(-90 12,{H // 2})">score</text>']
    for x, y, name, kind in pts:
        px = 50 + (W - 60) * x / xmax; py = H - 30 - (H - 40) * y
        sc.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{colors[kind]}"><title>{name}: {x:.1f} calls, {y:.3f}</title></circle><text x="{px + 7:.1f}" y="{py + 4:.1f}">{name}</text>')
    sc.append("</svg>"); f4 = "\n".join(sc)

    md = f"# Paper tables (generated {Path(__file__).name})\n\n## Table 1 — leaderboard\n\n{t1}\n\n## Table 2 — by category (coding agents)\n\n{t2}\n\n## Table 3 — efficiency and failure anatomy\n\n{t3}\n\n## Table 4 — hardest 15 tasks for the closed-model agents\n\n{t4}\n"
    (out / "tables.md").write_text(md)
    html = f"""<!doctype html><html><head><meta charset=utf-8><title>solve-vision-with-code · paper tables</title>
<style>body{{font:14px/1.45 -apple-system,Helvetica,Arial;margin:32px auto;max-width:1200px;color:#222}}table{{border-collapse:collapse;margin:8px 0 24px;font-size:13px}}td,th{{border:1px solid #ddd;padding:4px 8px;text-align:left}}th{{background:#f4f4f4}}h2{{margin-top:36px}}</style></head><body>
<h1>Coding agents on VBVR-Pro-Bench — paper tables and figures</h1>
<p>All scores from the unmodified official VBVR-Pro-Bench evaluator (kit-conforming environment). Unproduced instances count 0. Video-model rows are the published leaderboard numbers. CSVs next to this page.</p>
<h2>Figure 1 — overall score, all systems</h2>{f1}
<h2>Figure 2 — OOD minus ID (positive = generalises beyond the training families)</h2>{f2}
<h2>Figure 3 — per-task scores, closed-model agents (100 tasks, dark = 0, light = 1; hover for the task)</h2><div style="overflow-x:auto">{f3}</div>
<h2>Figure 4 — tool calls per instance vs score</h2>{f4}
{md_to_html(md)}
</body></html>"""
    (out / "paper_tables.html").write_text(html)
    print(f"wrote {out}/: tables.md, paper_tables.html, 4 CSVs")


def md_to_html(md: str) -> str:
    out = []
    for block in md.split("\n\n"):
        if block.startswith("#"):
            level = len(block.split(" ")[0]); out.append(f"<h{level + 1}>{block.lstrip('# ')}</h{level + 1}>")
        elif block.startswith("|"):
            lines = [l for l in block.splitlines() if not set(l) <= set("|-")]
            rows = [[c.strip() for c in l.strip("|").split("|")] for l in lines]
            out.append("<table><tr>" + "".join(f"<th>{c}</th>" for c in rows[0]) + "</tr>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows[1:]) + "</table>")
        else:
            out.append(f"<p>{block}</p>")
    return "\n".join(out)


if __name__ == "__main__":
    main()
