#!/usr/bin/env python3
"""Results figures and tables for the paper, generated from bench/paper/*.csv.

    .venv/bin/python paper/figures/make_figures.py            # writes the four files below and prints the checks
    .venv/bin/python paper/figures/make_figures.py --check    # checks only, writes nothing

Writes
    paper/figures/fig_leaderboard.pdf   overall score of all 37 systems with in/out-of-domain markers (FULL, \textwidth)
    paper/figures/fig_ood.pdf           in-domain -> out-of-domain dumbbell per system (HALF, beside prose)
    paper/tables/tab_main.tex           main leaderboard table
    paper/tables/tab_categories.tex     coding agents x five categories

Sources (read only): bench/paper/table1_leaderboard.csv, bench/paper/table2_categories.csv,
bench/results/<lane>/<lane>_vbvr_results.json (per-instance scores; the produced-video count comes from here),
bench/stats.json (cross-check of the produced counts). Video-model rows are the published VBVR-Pro leaderboard.

Style: paper/figures/STYLE.md (one-column ICML template, \textwidth 6.0 in; Times; INK/greys + one accent).
The rcParams block below is the one from STYLE.md section 4; make_analysis_figures.py imports it from here.
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

# ---------------------------------------------------------------- style (STYLE.md section 3-4, verbatim)
INK, GREY_1, GREY_2, GREY_3 = "#111111", "#555555", "#999999", "#dddddd"
AGENT, AGENT_OPEN, VIDEO = "#2a5db0", "#a9c0e6", "#a8a29a"
FULL, WIDE, HALF, WRAP = 6.0, 5.4, 2.9, 2.7          # inches, = 1.0 / 0.9 / 0.48 / 0.45 \textwidth

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],   # verified present on this Mac
    "mathtext.fontset": "stix",                 # Times-like math
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "axes.axisbelow": True,
    "lines.linewidth": 1.2, "lines.markersize": 4, "patch.linewidth": 0.6,
    "legend.frameon": False, "legend.handlelength": 1.2, "legend.borderaxespad": 0.2,
    "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,     # embed TrueType so Times survives in the PDF
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})

COLOR = {"closed": AGENT, "open": AGENT_OPEN, "video": VIDEO}
KIND_LABEL = {"closed": "Coding agent, closed model", "open": "Coding agent, open-weight model", "video": "Video model"}


def text_width(ax, s, **kw):
    """Width of the string in data units on this axes (measured with the figure's renderer, then removed)."""
    t = ax.text(0, 0, s, **kw)
    bb = t.get_window_extent(renderer=ax.figure.canvas.get_renderer())
    t.remove()
    inv = ax.transData.inverted()
    return inv.transform((bb.x1, 0))[0] - inv.transform((bb.x0, 0))[0]


# CSV model name -> paper display name (everything else keeps its CSV name)
DISPLAY = {
    "Codex/gpt-6-astra": "Codex (gpt-6-astra)",
    "Claude Code/Fable 5.1": "Claude Code (Fable 5.1)",
    "Gemini CLI/3.1 Pro": "Gemini CLI (Gemini 3.1 Pro)",
    "VBVR-Pro-Wan2.2-I2V-A14B (RL-trained video model)": "VBVR-Pro-Wan2.2-I2V-A14B (RL)",
}
# VBVR-Pro paper, Table 8: in-domain category means of the released baseline (the only video-model category numbers in the brief)
VIDEO_CAT_ID = {"model": "VBVR-Pro-Wan2.2-TI2V-5B", "Abstraction": 0.578, "Perception": 0.476, "Spatiality": 0.480, "Transformation": 0.724, "Knowledge": 0.511}


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
def is_trained(r):
    """Video models fine-tuned or RL-trained on VBVR / VBVR-Pro task families (the VBVR-* rows of the leaderboard)."""
    return r["kind"] == "video" and r["csv_name"].startswith("VBVR")


def fig_leaderboard(rows, path, n_values=6):
    """All 37 systems sorted by overall score: one horizontal bar per system coloured by family, the in-domain (hollow
    circle) and out-of-domain (solid diamond) scores as markers on the same row, a dashed rule at the best video model.
    Values at the bar end for the top n_values systems only (the rest are in Table 1). The four systems with no video in
    500 attempts (score 0) are listed in a footnote instead of drawn as empty bars.
    Columns: overall, in_domain, out_of_domain, kind of bench/paper/table1_leaderboard.csv."""
    shown = [r for r in rows if r["overall"] > 0]
    zeros = [r for r in rows if r["overall"] == 0]
    n = len(shown)
    fig, ax = plt.subplots(figsize=(FULL, 3.65))
    y = list(range(n))[::-1]
    for xv in (0.2, 0.4, 0.6, 0.8, 1.0):  # value-axis rules (the "horizontal gridlines" of a horizontal bar chart)
        ax.axvline(xv, color=GREY_3, linewidth=0.4, zorder=0)
    ax.barh(y, [r["overall"] for r in shown], height=0.72, color=[COLOR[r["kind"]] for r in shown], linewidth=0, zorder=2)
    for yi, r in zip(y, shown):
        ax.plot(r["ID"], yi, marker="o", markersize=3.2, markerfacecolor="white", markeredgecolor=INK, markeredgewidth=0.6, linestyle="none", zorder=4)
        ax.plot(r["OOD"], yi, marker="D", markersize=2.9, markerfacecolor=INK, markeredgecolor="white", markeredgewidth=0.5, linestyle="none", zorder=4)
    best_video = max((r for r in rows if r["kind"] == "video"), key=lambda r: r["overall"])
    ax.set_xlim(0, 1.06)
    ax.set_ylim(-0.7, n - 0.3)
    w = text_width(ax, "0.000", fontsize=7)
    for yi, r in list(zip(y, shown))[:n_values]:
        x = r["overall"] + 0.014
        for m in sorted((r["ID"], r["OOD"])):  # a marker under the text: step past it
            if x - 0.012 <= m <= x + w:
                x = m + 0.014
        if x - 0.012 <= best_video["overall"] <= x + w:  # the dashed rule under the text: step past it
            x = best_video["overall"] + 0.012
        ax.text(x, yi, f"{r['overall']:.3f}", va="center", ha="left", fontsize=7, color=INK)
    ax.axvline(best_video["overall"], color=GREY_2, linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
    ax.text(best_video["overall"] - 0.010, y[-1], f"best video model {best_video['overall']:.3f}", ha="right", va="center", fontsize=7, color=GREY_2)

    ax.set_yticks(y)
    ax.set_yticklabels([r["name"] for r in shown], fontsize=7)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.set_xlabel("Score (mean over 500 instances)", labelpad=3)
    ax.spines["left"].set_visible(False)

    fam = [Patch(color=COLOR[k], label=KIND_LABEL[k]) for k in ("closed", "open", "video")]
    mk = [Line2D([], [], marker="o", markersize=3.2, markerfacecolor="white", markeredgecolor=INK, markeredgewidth=0.6, linestyle="none", label="In-domain (50 tasks)"),
          Line2D([], [], marker="D", markersize=2.9, markerfacecolor=INK, markeredgecolor="white", markeredgewidth=0.5, linestyle="none", label="Out-of-domain (50 tasks)")]
    handles = [fam[0], mk[0], fam[1], mk[1], fam[2]]  # column-major fill -> row 1 the three families, row 2 the two markers
    leg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.04), ncol=3, fontsize=6.5, handletextpad=0.5, columnspacing=1.6, labelspacing=0.3, borderaxespad=0)
    for t in leg.get_texts():
        t.set_color(GREY_1)
    if zeros:
        fig.text(0.5, 0.0, "No video in any of the 500 attempts (score 0): " + ", ".join(r["name"] for r in zeros) + ".",
                 ha="center", va="bottom", fontsize=6.5, color=GREY_1)
    fig.subplots_adjust(left=0.245, right=0.99, top=0.995, bottom=0.20)
    fig.savefig(path)
    plt.close(fig)


def ood_group(r):
    return "agent" if r["kind"] != "video" else "trained" if is_trained(r) else "untrained"


# family colours; the untrained video models take a tint of VIDEO, as AGENT_OPEN is a tint of AGENT
VIDEO_UNTRAINED = "#d3cfc9"
OOD_COLOR = {"agent": AGENT, "trained": VIDEO, "untrained": VIDEO_UNTRAINED}
OOD_LABEL = {"agent": "Coding agent (no training on the benchmark)", "trained": "Video model trained on VBVR / VBVR-Pro families",
             "untrained": "Video model without benchmark training"}


def fig_ood(rows, path):
    """Dumbbell per system with at least one video: hollow dot at the in-domain score, solid dot at the out-of-domain
    score, connector between them; rows sorted by OOD - ID (gains at the top), every row named. Coding agents in the
    accent, video models trained on the benchmark's task families in the video grey, untrained video models in its tint.
    Columns: in_domain, out_of_domain, kind of bench/paper/table1_leaderboard.csv (the same numbers as Table 1)."""
    rows = [r for r in rows if r["kind"] == "video" or r["overall"] > 0]  # lanes with no video have no shift
    rows = sorted(rows, key=lambda r: r["OOD"] - r["ID"], reverse=True)
    n = len(rows)
    fig, ax = plt.subplots(figsize=(HALF, 3.55))
    y = list(range(n))[::-1]
    for xv in (0.2, 0.4, 0.6, 0.8, 1.0):
        ax.axvline(xv, color=GREY_3, linewidth=0.4, zorder=0)
    for yi, r in zip(y, rows):
        c = OOD_COLOR[ood_group(r)]
        ax.plot([r["ID"], r["OOD"]], [yi, yi], color=GREY_2, linewidth=0.8, solid_capstyle="butt", zorder=2)  # connector; the dots carry the family
        ax.plot(r["ID"], yi, marker="o", markersize=3.6, markerfacecolor="white", markeredgecolor=c, markeredgewidth=0.8, linestyle="none", zorder=3)
        ax.plot(r["OOD"], yi, marker="o", markersize=3.6, markerfacecolor=c, markeredgecolor=c, markeredgewidth=0.8, linestyle="none", zorder=4)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.set_yticks(y)
    ax.set_yticklabels([r["name"] for r in rows], fontsize=6)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.spines["left"].set_visible(False)
    fig.text(0.5, 0.145, "Score: in-domain (hollow) to out-of-domain (solid)", ha="center", va="bottom", fontsize=7, color=INK)  # centred on the figure: wider than the axes
    handles = [Line2D([], [], color=GREY_2, linewidth=0.8, marker="o", markersize=3.6, markerfacecolor=OOD_COLOR[g],
                      markeredgecolor=OOD_COLOR[g], label=OOD_LABEL[g]) for g in ("agent", "trained", "untrained")]
    leg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=1, fontsize=6.5, handletextpad=0.6, labelspacing=0.3, borderaxespad=0)
    for t in leg.get_texts():
        t.set_color(GREY_1)
    fig.subplots_adjust(left=0.46, right=0.985, top=0.995, bottom=0.215)
    fig.savefig(path)
    plt.close(fig)


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
        "\\begin{table}[!tp]", "\\centering",
        "\\caption{VBVR-Pro-Bench (video setting, 100 tasks $\\times$ 5 instances) leaderboard. Every system is scored by the unmodified official "
        "rule-based evaluator of the benchmark kit; the score is the mean over all 500 instances, and an instance for which the system produced no video "
        "counts 0. Coding-agent rows are our runs (one attempt per instance, sandboxed, no image or video model available); video-model rows are the published "
        "VBVR-Pro leaderboard numbers~\\citep{vbvrpro2026}, not rerun by us. \\emph{Rank} is the position among all 37 systems. \\emph{Videos} is the number of "
        "instances for which the agent wrote a video file (--- for video models, which we did not run). In-domain (ID) tasks are the 50 families with "
        "training data in VBVR-Pro, out-of-domain (OOD) the 50 families held out from all training. Best per column in bold, second underlined; the best "
        "coding agent and the best video model are shaded. "
        "$^\\dagger$No video in any of the 500 attempts (the model answers in prose and never calls a tool).}",
        "\\label{tab:main}", "\\footnotesize", "\\setlength{\\tabcolsep}{5pt}",
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
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
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
        "\\begin{table}[H]", "\\centering",
        "\\caption{Score by task category for the three closed-model agents and the eight best open-weight models (mean over the 5 instances of every "
        "task in the category, both splits; task counts per category in the second header row; best per column in bold, second underlined). The last row "
        "is the in-domain category profile of the released VBVR-Pro baseline video model as reported in the VBVR-Pro paper~\\citep[Table~8]{vbvrpro2026}, "
        "and because it covers the 50 in-domain tasks only, the block above it restricts the three closed-model agents to the same 50 tasks, computed from "
        "the per-instance results. Rows in the in-domain block are not ranked.}",
        "\\label{tab:categories}", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
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
    out.append(f"\\makecell[l]{{{v['model']}\\\\\\emph{{(video model)}}}} & " + " & ".join(f"{v[c]:.3f}" for c in CATS) + " & " + next(f"{x['ID']:.3f}" for x in rows if x["csv_name"] == v["model"]) + " \\\\")
    out += [
"\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
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
    fig_leaderboard(rows, FIG_DIR / "fig_leaderboard.pdf")
    fig_ood(rows, FIG_DIR / "fig_ood.pdf")
    tab_main(rows, produced, TAB_DIR / "tab_main.tex")
    tab_categories(cats, rows, n_tasks, n_tasks_id, TAB_DIR / "tab_categories.tex")
    print(f"wrote fig_leaderboard.pdf (FULL {FULL} in), fig_ood.pdf (HALF {HALF} in), tab_main.tex, tab_categories.tex")


if __name__ == "__main__":
    main()
