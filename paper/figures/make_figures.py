#!/usr/bin/env python3
"""Results figures and tables for the paper, generated from bench/paper/*.csv.

    .venv/bin/python paper/figures/make_figures.py            # writes the four files below and prints the checks
    .venv/bin/python paper/figures/make_figures.py --check    # checks only, writes nothing

Writes
    paper/figures/fig_leaderboard.pdf   overall score of all 37 systems + in/out-of-domain dumbbells (two-column)
    paper/figures/fig_ood.pdf           out-of-domain minus in-domain per system (single column)
    paper/tables/tab_main.tex           main leaderboard table (table*)
    paper/tables/tab_categories.tex     coding agents x five categories

Sources (read only): bench/paper/table1_leaderboard.csv, bench/paper/table2_categories.csv,
bench/results/<lane>/<lane>_vbvr_results.json (per-instance scores; the produced-video count comes from here),
bench/stats.json (cross-check of the produced counts). Video-model rows are the published VBVR-Pro leaderboard.
"""
import argparse, csv, json, sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
from report import LANES, ZERO_LANES, CATS  # noqa: E402
from paper_tables import short, CLOSED  # noqa: E402

CSV_DIR = ROOT / "bench" / "paper"
FIG_DIR = ROOT / "paper" / "figures"
TAB_DIR = ROOT / "paper" / "tables"

SINGLE, DOUBLE = 3.25, 6.75  # ICML column widths (in)
COLOR = {"closed": "#2a78d6", "open": "#1baf7a", "video": "#eb6834"}  # validated categorical triple
KIND_LABEL = {"closed": "Coding agent, closed model", "open": "Coding agent, open-weight model (OpenCode)", "video": "Video model (VBVR-Pro leaderboard)"}

# CSV model name -> paper display name (everything else keeps its CSV name)
DISPLAY = {
    "Codex/gpt-6-astra": "Codex (gpt-6-astra)",
    "Claude Code/Fable 5.1": "Claude Code (Fable 5.1)",
    "Gemini CLI/3.1 Pro": "Gemini CLI (Gemini 3.1 Pro)",
    "VBVR-Pro-Wan2.2-I2V-A14B (RL-trained video model)": "VBVR-Pro-Wan2.2-I2V-A14B (RL)",
}
# VBVR-Pro paper, Table 8: in-domain category means of the released baseline (the only video-model category numbers in the brief)
VIDEO_CAT_ID = {"model": "VBVR-Pro-Wan2.2-TI2V-5B", "Abstraction": 0.578, "Perception": 0.476, "Spatiality": 0.480, "Transformation": 0.724, "Knowledge": 0.511}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 7,
    "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
})


# ---------------------------------------------------------------- data
def kind_of(kind_str):
    return "video" if kind_str == "video model" else "open" if "open" in kind_str else "closed"


def name_of(csv_name):
    if csv_name in DISPLAY:
        return DISPLAY[csv_name]
    if csv_name.startswith("OpenCode × "):  # zero-video lanes carry the long report.py name
        return csv_name.replace("OpenCode × ", "").split(" (")[0]
    return csv_name


def load_leaderboard():
    rows = []
    with open(CSV_DIR / "table1_leaderboard.csv") as f:
        for r in csv.DictReader(f):
            rows.append(dict(csv_name=r["model"], name=name_of(r["model"]), kind=kind_of(r["kind"]),
                             overall=float(r["overall"]), ID=float(r["in_domain"]), OOD=float(r["out_of_domain"])))
    rows.sort(key=lambda r: -r["overall"])
    return rows


def load_categories():
    with open(CSV_DIR / "table2_categories.csv") as f:
        return [dict(name=name_of(r["model"]), csv_name=r["model"], overall=float(r["overall"]), **{c: float(r[c]) for c in CATS}) for r in csv.DictReader(f)]


def produced_counts():
    """Videos produced per lane, from the per-instance results (an unproduced instance is scored 0 with error 'no prediction')."""
    out = {}
    for lane in LANES:
        p = ROOT / "bench" / "results" / lane / f"{lane}_vbvr_results.json"
        if p.exists():
            out[short(lane)] = sum(1 for x in json.load(open(p))["samples"] if x["error"] != "no prediction")
    for lane, name in ZERO_LANES.items():
        out[name] = 0  # keyed like the CSV
    return out


def per_instance(lane):
    return json.load(open(ROOT / "bench" / "results" / lane / f"{lane}_vbvr_results.json"))["samples"]


# ---------------------------------------------------------------- figures
def fig_leaderboard(rows, path):
    n = len(rows)
    fig_h = 0.155 * n + 0.75
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(DOUBLE, fig_h), sharey=True, gridspec_kw=dict(width_ratios=[1.45, 1.0], wspace=0.06))
    y = list(range(n))[::-1]
    cols = [COLOR[r["kind"]] for r in rows]
    ax.barh(y, [r["overall"] for r in rows], height=0.72, color=cols, linewidth=0)
    for yi, r in zip(y, rows):
        ax.text(r["overall"] + 0.012, yi, f"{r['overall']:.3f}", va="center", ha="left", fontsize=6.5, color="#333")
    ax.set_yticks(y)
    ax.set_yticklabels([r["name"] + (" $^\\dagger$" if r["overall"] == 0 else "") for r in rows])
    ax.set_xlim(0, 1.08)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Overall score (mean over 500 instances)")
    ax.set_ylim(-0.7, n - 0.3)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.4)
    ax.set_axisbelow(True)

    # dumbbell: hollow = in-domain, filled = out-of-domain
    for yi, r in zip(y, rows):
        c = COLOR[r["kind"]]
        bx.plot([r["ID"], r["OOD"]], [yi, yi], color=c, linewidth=1.0, solid_capstyle="round", zorder=1)
        bx.plot(r["ID"], yi, marker="o", markersize=3.6, markerfacecolor="white", markeredgecolor=c, markeredgewidth=0.9, zorder=2)
        bx.plot(r["OOD"], yi, marker="o", markersize=3.6, markerfacecolor=c, markeredgecolor=c, markeredgewidth=0.9, zorder=3)
    bx.set_xlim(-0.02, 1.02)
    bx.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    bx.set_xlabel("In-domain (hollow) vs. out-of-domain (filled)")
    bx.tick_params(axis="y", length=0)
    bx.grid(axis="x", color="#e5e5e5", linewidth=0.4)
    bx.set_axisbelow(True)

    handles = [Patch(color=COLOR[k], label=KIND_LABEL[k]) for k in ("closed", "open", "video")]
    handles += [Line2D([], [], marker="o", color="#555", markerfacecolor="white", markersize=3.6, linewidth=0, label="In-domain (50 tasks)"),
                Line2D([], [], marker="o", color="#555", markerfacecolor="#555", markersize=3.6, linewidth=0, label="Out-of-domain (50 tasks)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=1.2, columnspacing=1.2)
    fig.subplots_adjust(left=0.235, right=0.99, top=0.995, bottom=0.72 / fig_h)
    fig.savefig(path)
    plt.close(fig)
    return fig_h


def is_trained(r):
    """Video models fine-tuned or RL-trained on VBVR / VBVR-Pro task families (the VBVR-* rows of the leaderboard)."""
    return r["kind"] == "video" and r["csv_name"].startswith("VBVR")


def fig_ood(rows, path):
    rows = [r for r in rows if r["kind"] == "video" or r["overall"] > 0]  # lanes with no video have no delta
    rows = sorted(rows, key=lambda r: r["OOD"] - r["ID"], reverse=True)
    n = len(rows)
    fig_h = 0.125 * n + 1.05
    fig, ax = plt.subplots(figsize=(SINGLE, fig_h))
    y = list(range(n))[::-1]
    d = [r["OOD"] - r["ID"] for r in rows]
    # three groups: coding agents (blue), VBVR-trained video models (solid orange), untrained video models (hatched orange)
    for yi, r, di in zip(y, rows, d):
        if r["kind"] != "video":
            ax.barh(yi, di, height=0.72, color=COLOR["closed"], linewidth=0)
        elif is_trained(r):
            ax.barh(yi, di, height=0.72, color=COLOR["video"], linewidth=0)
        else:
            ax.barh(yi, di, height=0.72, facecolor="white", edgecolor=COLOR["video"], linewidth=0.6, hatch="//////")
        ax.text(di + (0.006 if di >= 0 else -0.006), yi, f"{di:+.3f}", va="center", ha="left" if di >= 0 else "right", fontsize=6, color="#333")
    ax.axvline(0, color="#222", linewidth=0.6, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([r["name"] for r in rows], fontsize=6.5)
    ax.tick_params(axis="y", length=0)
    lo, hi = min(d), max(d)
    ax.set_xlim(lo - 0.10, hi + 0.10)
    ax.set_xlabel("Out-of-domain minus in-domain score")
    ax.set_ylim(-0.7, n - 0.3)
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.4)
    ax.set_axisbelow(True)
    handles = [Patch(color=COLOR["closed"], label="Coding agent (no training on the benchmark)"),
               Patch(color=COLOR["video"], label="Video model trained on VBVR / VBVR-Pro task families"),
               Patch(facecolor="white", edgecolor=COLOR["video"], hatch="//////", linewidth=0.6, label="Video model without benchmark training")]
    fig.legend(handles=handles, loc="lower left", ncol=1, frameon=False, handlelength=1.2, bbox_to_anchor=(0.02, 0.0), borderaxespad=0.0)
    fig.subplots_adjust(left=0.44, right=0.985, top=0.995, bottom=0.92 / fig_h)
    fig.savefig(path)
    plt.close(fig)
    return fig_h


# ---------------------------------------------------------------- tables
def fmt(v, best, second):
    s = f"{v:.3f}"
    return f"\\textbf{{{s}}}" if v == best else f"\\underline{{{s}}}" if v == second else s


def top2(vals):
    u = sorted(set(vals), reverse=True)
    return u[0], (u[1] if len(u) > 1 else None)


def tex_name(name):
    return name.replace("×", "$\\times$").replace("_", "\\_")


def tab_main(rows, produced, path):
    groups = [("closed", "Coding agents, closed models"), ("open", "Coding agents, open-weight models (OpenCode)"), ("video", "Video models (VBVR-Pro leaderboard)")]
    bests = {c: top2([r[c] for r in rows]) for c in ("overall", "ID", "OOD")}
    rank = {r["csv_name"]: i for i, r in enumerate(rows, 1)}
    best_video = max((r for r in rows if r["kind"] == "video"), key=lambda r: r["overall"])
    out = [
        "% Generated by paper/figures/make_figures.py from bench/paper/table1_leaderboard.csv and bench/results/ -- do not edit by hand.",
        "\\begin{table*}[t]", "\\centering",
        "\\caption{VBVR-Pro-Bench (video setting, 100 tasks $\\times$ 5 instances) leaderboard. Every system is scored by the unmodified official "
        "rule-based evaluator of the benchmark kit; the score is the mean over all 500 instances, and an instance for which the system produced no video "
        "counts 0. Coding-agent rows are our runs (one attempt per instance, sandboxed, no image or video model available); video-model rows are the published "
        "VBVR-Pro leaderboard numbers~\\citep{xu2026vbvrpro}, not rerun by us. \\emph{Rank} is the position among all 37 systems. \\emph{Videos} is the number of "
        "instances for which the agent wrote a video file (--- for video models, which we did not run). In-domain (ID) tasks are the 50 families with "
        "training data in VBVR-Pro, out-of-domain (OOD) the 50 families held out from all training. Best per column in bold, second underlined; the best "
        "coding agent and the best video model are shaded. "
        "$^\\dagger$No video in any of the 500 attempts (the model answers in prose and never calls a tool).}",
        "\\label{tab:main}", "\\small", "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{rlcccc}", "\\toprule",
        "Rank & System & Overall & ID & OOD & Videos / 500 \\\\",
    ]
    for kind, title in groups:
        out += ["\\midrule", f"\\multicolumn{{6}}{{l}}{{\\emph{{{title}}}}} \\\\[1pt]"]
        for r in [r for r in rows if r["kind"] == kind]:
            shade = "\\rowcolor{bestcolor} " if r is rows[0] else "\\rowcolor{secondcolor} " if r is best_video else ""
            if kind == "video":
                prod = "---"
            else:
                p = produced[r["csv_name"]]
                prod = f"{p}$^\\dagger$" if p == 0 else str(p)
            cells = [fmt(r[c], *bests[c]) for c in ("overall", "ID", "OOD")]
            out.append(f"{shade}{rank[r['csv_name']]} & {tex_name(r['name'])} & " + " & ".join(cells) + f" & {prod} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    path.write_text("\n".join(out))


def id_only_categories(lane):
    s = per_instance(lane)
    return {c: sum(x["score"] for x in s if x["category"] == c and x["split"] == "In_Domain") / sum(1 for x in s if x["category"] == c and x["split"] == "In_Domain") for c in CATS}


def tab_categories(cats, rows, n_tasks, n_tasks_id, path, n_open=8):
    closed = [c for c in cats if c["csv_name"] in ("Codex/gpt-6-astra", "Claude Code/Fable 5.1", "Gemini CLI/3.1 Pro")]
    opens = sorted([c for c in cats if c not in closed], key=lambda c: -c["overall"])[:n_open]
    shown = closed + opens
    cols = CATS + ["overall"]
    bests = {c: top2([r[c] for r in shown]) for c in cols}
    head = " & ".join(CATS)
    counts = " & ".join(f"({n_tasks[c]})" for c in CATS)
    out = [
        "% Generated by paper/figures/make_figures.py from bench/paper/table2_categories.csv -- do not edit by hand.",
        "\\begin{table*}[t]", "\\centering",
        "\\caption{Coding-agent score by task category (mean over the 5 instances of every task in the category, both splits; the number of tasks per "
        "category is in the second header row). The three closed-model agents and the eight best open-weight models are shown; best per column in bold, second "
        "underlined. The last row is the in-domain category profile of the released VBVR-Pro baseline video model as reported in the VBVR-Pro "
        "paper~\\citep[Table~8]{xu2026vbvrpro}; because it covers the 50 in-domain tasks only, the block above it restricts the three closed-model "
        "agents to the same 50 tasks (task counts per category in that block's header row; computed from the per-instance results). "
        "Rows in the in-domain block are not ranked.}",
        "\\label{tab:categories}", "\\small", "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lcccccc}", "\\toprule",
        f"System & {head} & Overall \\\\", f"\\emph{{Tasks per category}} & {counts} & (100) \\\\", "\\midrule",
        "\\multicolumn{7}{l}{\\emph{Coding agents, closed models}} \\\\[1pt]",
    ]
    for i, r in enumerate(shown):
        if i == len(closed):
            out += ["\\midrule", "\\multicolumn{7}{l}{\\emph{Coding agents, open-weight models (OpenCode)}} \\\\[1pt]"]
        out.append(f"{tex_name(r['name'])} & " + " & ".join(fmt(r[c], *bests[c]) for c in cols) + " \\\\")
    v = VIDEO_CAT_ID
    idn = " & ".join(f"({n_tasks_id[c]})" for c in CATS)
    out += ["\\midrule", f"\\multicolumn{{7}}{{l}}{{\\emph{{In-domain tasks only}}}} \\\\[1pt]", f"\\emph{{Tasks per category}} & {idn} & (50) \\\\"]
    for lane in CLOSED:
        r = next(c for c in cats if c["csv_name"] == short(lane)); idc = id_only_categories(lane)
        idov = next(x for x in rows if x["csv_name"] == r["csv_name"])["ID"]
        out.append(f"{tex_name(r['name'])} & " + " & ".join(f"{idc[c]:.3f}" for c in CATS) + f" & {idov:.3f} \\\\")
    out.append(f"{v['model']} (video model) & " + " & ".join(f"{v[c]:.3f}" for c in CATS) + " & " + next(f"{x['ID']:.3f}" for x in rows if x["csv_name"] == v["model"]) + " \\\\")
    out += [
"\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    path.write_text("\n".join(out))
    return shown


# ---------------------------------------------------------------- checks
def checks(rows, cats, produced):
    print("== leaderboard (37 rows, sorted by overall) ==")
    for i, r in enumerate(rows, 1):
        print(f"{i:2d} {r['kind']:6s} {r['name']:36s} {r['overall']:.4f} {r['ID']:.4f} {r['OOD']:.4f} d={r['OOD'] - r['ID']:+.4f}"
              + (f" produced={produced[r['csv_name']]}" if r["kind"] != "video" else ""))
    agents = [r for r in rows if r["kind"] != "video" and r["overall"] > 0]
    videos = [r for r in rows if r["kind"] == "video"]
    assert len(rows) == 37 and len(videos) == 11 and len(agents) == 22 and sum(1 for r in rows if r["overall"] == 0) == 4
    pos_agents = [r for r in agents if r["OOD"] > r["ID"]]
    neg_agents = [(r["name"], produced[r["csv_name"]], round(r["OOD"] - r["ID"], 4)) for r in agents if r["OOD"] <= r["ID"]]
    print(f"OOD-ID > 0 for {len(pos_agents)}/{len(agents)} coding agents with a video; exceptions:", neg_agents)
    trained = [r for r in videos if r["csv_name"].startswith("VBVR")]
    print(f"OOD-ID < 0 for {sum(r['OOD'] < r['ID'] for r in trained)}/{len(trained)} VBVR-trained video models:", [(r["name"], round(r["OOD"] - r["ID"], 3)) for r in trained])
    print("other video models:", [(r["name"], round(r["OOD"] - r["ID"], 3)) for r in videos if r not in trained])
    assert len(neg_agents) == 1 and neg_agents[0][0] == "Qwen3-32B", "brief claim: the only coding-agent exception is Qwen3-32B"
    assert all(r["OOD"] < r["ID"] for r in trained)
    best_open = max((r for r in agents if r["kind"] == "open"), key=lambda r: r["overall"])
    beaten = [v["name"] for v in videos if v["overall"] < best_open["overall"]]
    print(f"best open {best_open['name']} {best_open['overall']:.4f} beats {len(beaten)}/11 video models; not beaten:", [v["name"] for v in videos if v["overall"] >= best_open["overall"]])
    # produced counts vs bench/stats.json
    st = json.load(open(ROOT / "bench" / "stats.json"))
    mism = [(short(l), produced.get(short(l)), s["produced"]) for l, s in st.items() if l in LANES and produced.get(short(l)) != s["produced"]]
    print("produced counts agree with bench/stats.json:", not mism, mism)
    # category table numbers vs overall in leaderboard
    ov = {r["csv_name"]: r["overall"] for r in rows}
    assert all(abs(c["overall"] - ov[c["csv_name"]]) < 1e-12 for c in cats)
    print("category-table overall == leaderboard overall for all", len(cats), "agents: True")
    # category means recomputed from per-instance results for the closed three (and their in-domain-only profile)
    for lane in CLOSED:
        s = per_instance(lane)
        row = next(c for c in cats if c["csv_name"] == short(lane))
        rec = {c: sum(x["score"] for x in s if x["category"] == c) / sum(1 for x in s if x["category"] == c) for c in CATS}
        assert all(abs(rec[c] - row[c]) < 1e-9 for c in CATS)
        idc = {c: sum(x["score"] for x in s if x["category"] == c and x["split"] == "In_Domain") / sum(1 for x in s if x["category"] == c and x["split"] == "In_Domain") for c in CATS}
        print(f"{short(lane):24s} category means match CSV: True | ID-only: " + " ".join(f"{c[:5]} {idc[c]:.3f}" for c in CATS))
    # per-column best / second in the category table
    shown = [c for c in cats if c["csv_name"] in ("Codex/gpt-6-astra", "Claude Code/Fable 5.1", "Gemini CLI/3.1 Pro")] + sorted([c for c in cats if c["csv_name"] not in ("Codex/gpt-6-astra", "Claude Code/Fable 5.1", "Gemini CLI/3.1 Pro")], key=lambda c: -c["overall"])[:8]
    for c in CATS + ["overall"]:
        b = sorted(shown, key=lambda r: -r[c])
        print(f"  {c:15s} best {b[0]['name']} {b[0][c]:.3f}; second {b[1]['name']} {b[1][c]:.3f}; best open {next(r for r in b if r['csv_name'] not in ('Codex/gpt-6-astra', 'Claude Code/Fable 5.1', 'Gemini CLI/3.1 Pro'))['name']}")
    print("agent mean over closed three:", sum(r["overall"] for r in rows[:3]) / 3)
    print("gap codex - best video:", rows[0]["overall"] - max(v["overall"] for v in videos))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="print the checks only, write nothing")
    a = ap.parse_args()
    rows, cats, produced = load_leaderboard(), load_categories(), produced_counts()
    n_tasks, n_tasks_id = {}, {}
    for x in per_instance(CLOSED[0]):
        n_tasks.setdefault(x["category"], set()).add(x["task_name"])
        if x["split"] == "In_Domain":
            n_tasks_id.setdefault(x["category"], set()).add(x["task_name"])
    n_tasks = {c: len(v) for c, v in n_tasks.items()}; n_tasks_id = {c: len(v) for c, v in n_tasks_id.items()}
    checks(rows, cats, produced)
    print("tasks per category:", n_tasks, "in-domain only:", n_tasks_id)
    if a.check:
        return
    FIG_DIR.mkdir(exist_ok=True); TAB_DIR.mkdir(exist_ok=True)
    h1 = fig_leaderboard(rows, FIG_DIR / "fig_leaderboard.pdf")
    h2 = fig_ood(rows, FIG_DIR / "fig_ood.pdf")
    tab_main(rows, produced, TAB_DIR / "tab_main.tex")
    tab_categories(cats, rows, n_tasks, n_tasks_id, TAB_DIR / "tab_categories.tex")
    print(f"wrote fig_leaderboard.pdf ({DOUBLE} x {h1:.2f} in), fig_ood.pdf ({SINGLE} x {h2:.2f} in), tab_main.tex, tab_categories.tex")


if __name__ == "__main__":
    main()
