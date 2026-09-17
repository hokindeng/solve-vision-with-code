#!/usr/bin/env python3
"""Analysis figures for the paper (Section 5).

    .venv/bin/python paper/figures/make_analysis_figures.py            # writes the two PDFs and prints the checks
    .venv/bin/python paper/figures/make_analysis_figures.py --check    # checks only, writes nothing; exit 1 on a mismatch

Reads only the paper CSVs and the evaluator results:
  bench/paper/table3_efficiency.csv      -> fig_tools_vs_score.pdf  (columns score, tool_calls_per_attempt, produced, timeouts)
  bench/paper/table4_per_task_closed.csv -> fig_heatmap.pdf         (task, split, one column per closed agent)
  bench/results/codex-gpt-6-astra/*_vbvr_results.json (task -> category)
Checks (--check): table3 score == table1 overall per model, table3 produced == bench/stats.json produced, the mean of the
100 per-task scores in table4 == table1 overall for each closed agent, and the split labels are the two expected ones.

Style: paper/figures/STYLE.md; the rcParams block, palette and widths are imported from make_figures.py so the two
scripts cannot drift. HALF (2.9 in) for the scatter beside prose, FULL (6.0 in) for the heatmap.
"""
import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import NullFormatter
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from make_figures import INK, GREY_1, GREY_2, GREY_3, AGENT, AGENT_OPEN, FULL, HALF, text_width  # noqa: E402
from paper_tables import short, CLOSED as CLOSED_LANES  # noqa: E402  (bench/ is on sys.path after importing make_figures)

CLOSED = {short(l): n for l, n in zip(CLOSED_LANES, ("Codex", "Claude Code", "Gemini CLI"))}  # CSV name -> short label
CATS = ["Abstraction", "Perception", "Spatiality", "Transformation", "Knowledge"]
SPLITS = [("In_Domain", "In-domain (50 tasks)"), ("Out_of_Domain", "Out-of-domain (50 tasks)")]
HARD_MEAN = 0.7  # tasks whose three-agent mean is below this are named under the heatmap (Section 5: "on 13 the three-agent mean is below 0.7")


def load_efficiency():
    rows = list(csv.DictReader(open(ROOT / "bench/paper/table3_efficiency.csv")))
    for r in rows:
        r["x"], r["y"], r["n"] = float(r["tool_calls_per_attempt"]), float(r["score"]), int(r["produced"])
        r["closed"] = r["model"] in CLOSED
    return rows


def load_per_task():
    return list(csv.DictReader(open(ROOT / "bench/paper/table4_per_task_closed.csv")))


# ---------------------------------------------------------------- score vs tool calls
def fig_tools_vs_score(rows):
    """One point per lane that produced at least one video: x = mean tool calls per instance (log), y = score, area
    proportional to the number of produced videos (out of 500). Labels: the three closed agents, the three best
    open-weight lanes, the lane with the most timeouts and the two lanes that give up at the origin (fewest calls)."""
    rows = [r for r in rows if r["n"] > 0]
    opens = sorted((r for r in rows if not r["closed"]), key=lambda r: -r["y"])
    most_timeouts = max(rows, key=lambda r: int(r["timeouts"]))
    fewest = sorted(rows, key=lambda r: r["x"])[:2]  # the two "give up" lanes, on top of each other at the origin
    labelled = [r for r in rows if r["closed"]] + opens[:3] + [most_timeouts]

    fig, ax = plt.subplots(figsize=(HALF, 2.45))
    for r in sorted(rows, key=lambda r: -r["n"]):  # small discs drawn last so they stay visible
        ax.scatter(r["x"], r["y"], s=4 + 40 * r["n"] / 500, facecolor=AGENT if r["closed"] else AGENT_OPEN,
                   edgecolor=INK if r["closed"] else GREY_1, linewidth=0.4, zorder=3)
    # labels: hand offsets in points (adjustText is not a dependency of the repo)
    off = {"Codex": (0, 6, "center", "bottom"), "Claude Code": (4, -5, "left", "top"), "Gemini CLI": (5, 0, "left", "center"),
           "GLM-5": (4, 4, "left", "bottom"), "Kimi K2.5": (-5, 2, "right", "center"), "MiniMax M2.5": (5, -1, "left", "center"),
           "Qwen3-Coder-Next": (5, 0, "left", "center")}
    for r in labelled:
        name = CLOSED.get(r["model"], r["model"])
        dx, dy, ha, va = off[name]
        text = name + (f"\n({r['timeouts']} timeouts)" if r is most_timeouts else "")
        ax.annotate(text, (r["x"], r["y"]), xytext=(dx, dy), textcoords="offset points", fontsize=6.5, ha=ha, va=va, color=INK, zorder=4)
    fx, fy = fewest[0]["x"], fewest[0]["y"]
    ax.annotate(", ".join(r["model"] for r in fewest), (fx, fy), xytext=(fx * 1.35, 0.17), textcoords="data", fontsize=6.5,
                ha="left", va="center", color=INK, arrowprops=dict(arrowstyle="-", color=GREY_2, linewidth=0.5, shrinkA=0, shrinkB=2.5), zorder=4)

    ax.set_xscale("log")
    ax.set_xlim(1.4, 320)
    ax.set_ylim(-0.03, 1.02)
    ax.set_xticks([2, 5, 10, 20, 50, 100, 200])
    ax.set_xticklabels(["2", "5", "10", "20", "50", "100", "200"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", length=1.5)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.set_xlabel("Tool calls per instance (mean, log scale)", labelpad=3)
    ax.set_ylabel("Score", labelpad=3)
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=4.2, markerfacecolor=AGENT, markeredgecolor=INK, markeredgewidth=0.4, label="Coding agent, closed model"),
               Line2D([], [], linestyle="none", marker="o", markersize=4.2, markerfacecolor=AGENT_OPEN, markeredgecolor=GREY_1, markeredgewidth=0.4, label="Coding agent, open-weight model"),
               Line2D([], [], linestyle="none", marker="o", markersize=2.2, markerfacecolor="white", markeredgecolor=GREY_1, markeredgewidth=0.4, label="Area $\\propto$ videos produced / 500")]
    leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.80), fontsize=6.5, handletextpad=0.4, labelspacing=0.3, borderaxespad=0.2)
    for t in leg.get_texts():
        t.set_color(GREY_1)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.985, bottom=0.15)
    fig.savefig(OUT / "fig_tools_vs_score.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- per-task heatmap
def task_categories():
    f = glob.glob(str(ROOT / "bench/results/codex-gpt-6-astra/*_vbvr_results.json"))[0]
    cat = {}
    for s in json.load(open(f))["samples"]:
        cat[s["task_name"].replace("_data-generator", "")] = s["category"]
    return cat


def spread(xs, gap, lo, hi, iters=200):
    """Push sorted positions apart until neighbours are >= gap apart, staying inside [lo, hi]; deterministic."""
    xs = [float(x) for x in xs]
    for _ in range(iters):
        moved = False
        for k in range(len(xs) - 1):
            d = xs[k + 1] - xs[k]
            if d < gap - 1e-9:
                xs[k] -= (gap - d) / 2; xs[k + 1] += (gap - d) / 2; moved = True
        xs = [min(max(x, lo), hi) for x in xs]
        if not moved:
            break
    return xs


def fig_heatmap(rows):
    """3 agent rows x 100 task columns, in-domain block left and out-of-domain block right. Within a block tasks are
    grouped by category (band above, labels on alternating rows so narrow bands never overprint) and sorted by the
    three-agent mean, ascending. One single-hue ramp, white to the accent. No per-task ticks: tasks whose three-agent
    mean is below HARD_MEAN are named under the axis with a thin leader; every other task id is omitted."""
    cat = task_categories()
    agents = list(CLOSED)
    cmap = LinearSegmentedColormap.from_list("accent", ["#ffffff", AGENT])
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.2), gridspec_kw={"wspace": 0.05, "width_ratios": [50, 50]})
    for ax, (split, title) in zip(axes, SPLITS):
        sub = [r for r in rows if r["split"] == split]
        sub.sort(key=lambda r: (CATS.index(cat[r["task"]]), np.mean([float(r[a]) for a in agents])))
        M = np.array([[float(r[a]) for r in sub] for a in agents])
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        # row borders in paper white; no per-cell vertical borders (a 3.5 pt cell with a 0.5 pt border reads as a moire)
        for k in range(len(agents) + 1):
            ax.axhline(k - 0.5, color="white", linewidth=0.5)
        ax.set_yticks(range(len(agents)))
        ax.set_yticklabels([CLOSED[a] for a in agents], fontsize=7, color=GREY_1)
        ax.tick_params(axis="y", length=0, pad=3)
        ax.set_xticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        # category bands above the block, labels alternating between two rows
        start = 0
        for k, c in enumerate(CATS):
            n = sum(1 for r in sub if cat[r["task"]] == c)
            if n == 0:
                continue
            ax.plot([start - 0.35, start + n - 0.65], [-0.72, -0.72], color=GREY_1, linewidth=0.6, clip_on=False, solid_capstyle="butt")
            ax.text(start + n / 2 - 0.5, -0.9 - 0.4 * (k % 2), c, ha="center", va="bottom", fontsize=6.5, color=GREY_1, clip_on=False)
            if start > 0:
                ax.axvline(start - 0.5, color="white", linewidth=1.6)
            start += n
        # the hard tasks, named under the axis with a leader to the column
        hard = [i for i, r in enumerate(sub) if np.mean([float(r[a]) for a in agents]) < HARD_MEAN]
        gap = 1.15 * text_width(ax, "G", fontsize=5.5) if hard else 0  # a rotated 5.5 pt label is one cap-height wide
        for i, x in zip(hard, spread(hard, gap, -0.5, len(sub) - 0.5)):
            ax.plot([i, x], [2.55, 2.92], color=GREY_2, linewidth=0.4, clip_on=False, zorder=1)
            ax.text(x, 2.98, sub[i]["task"].split("_")[0], rotation=90, ha="center", va="top", fontsize=5.5, color=INK, clip_on=False)
        ax.text(len(sub) / 2 - 0.5, 3.9, title, ha="center", va="top", fontsize=7.5, color=GREY_1, clip_on=False)  # block title under the task names
        ax.set_xlim(-0.5, len(sub) - 0.5)
        ax.set_ylim(len(agents) - 0.5, -0.5)  # the bands and leaders are drawn outside; keep the cells flush with the axes
    axes[1].tick_params(axis="y", labelleft=False)
    fig.subplots_adjust(left=0.085, right=0.895, top=0.82, bottom=0.31)
    cax = fig.add_axes([0.912, 0.31, 0.009, 0.51])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 0.5, 1.0])
    cb.set_ticklabels(["0.000", "0.500", "1.000"])
    cb.set_label("mean score over 5 instances", fontsize=6.5, labelpad=3, color=GREY_1)
    cb.ax.tick_params(labelsize=6.5, width=0.4, length=2, color=GREY_1)
    cb.outline.set_linewidth(0.4)
    cb.outline.set_edgecolor(GREY_1)
    fig.savefig(OUT / "fig_heatmap.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- checks
def checks(eff, per_task):
    """Recompute what the two figures draw against the other sources of record; return the list of mismatches."""
    bad = []
    lead = {r["model"]: r for r in csv.DictReader(open(ROOT / "bench/paper/table1_leaderboard.csv"))}
    stats = json.load(open(ROOT / "bench/stats.json"))
    stats_by_name = {short(lane): s for lane, s in stats.items()}
    for r in eff:
        if r["model"] in lead and abs(r["y"] - float(lead[r["model"]]["overall"])) > 1e-9:
            bad.append(f"table3 score {r['y']} != table1 overall {lead[r['model']]['overall']} for {r['model']}")
        if r["model"] in stats_by_name and r["n"] != stats_by_name[r["model"]]["produced"]:
            bad.append(f"table3 produced {r['n']} != stats.json {stats_by_name[r['model']]['produced']} for {r['model']}")
    splits = {r["split"] for r in per_task}
    if splits != {s for s, _ in SPLITS} or len(per_task) != 100:
        bad.append(f"table4: {len(per_task)} rows, splits {sorted(splits)}")
    for a in CLOSED:
        m = np.mean([float(r[a]) for r in per_task])
        if abs(m - float(lead[a]["overall"])) > 1e-9:
            bad.append(f"table4 mean over 100 tasks {m} != table1 overall {lead[a]['overall']} for {a}")
    n_hard = sum(1 for r in per_task if np.mean([float(r[a]) for a in CLOSED]) < HARD_MEAN)
    print(f"lanes with a video: {sum(1 for r in eff if r['n'] > 0)}; tasks with three-agent mean < {HARD_MEAN}: {n_hard}")
    print("checks:", "all agree" if not bad else "\n  ".join(["MISMATCH"] + bad))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="print the checks only, write nothing; exit 1 on a mismatch")
    a = ap.parse_args()
    eff, per_task = load_efficiency(), load_per_task()
    bad = checks(eff, per_task)
    if a.check:
        sys.exit(1 if bad else 0)
    fig_tools_vs_score(eff)
    fig_heatmap(per_task)
    print("wrote", OUT / "fig_tools_vs_score.pdf", OUT / "fig_heatmap.pdf")


if __name__ == "__main__":
    main()
