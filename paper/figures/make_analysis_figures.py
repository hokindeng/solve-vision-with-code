#!/usr/bin/env python3
"""Analysis figures for the paper (Section 5).

    .venv/bin/python paper/figures/make_analysis_figures.py

Reads only the paper CSVs and the evaluator results:
  bench/paper/table3_efficiency.csv      -> fig_tools_vs_score.pdf
  bench/paper/table4_per_task_closed.csv -> fig_heatmap.pdf
  bench/results/codex-gpt-6-astra/*_vbvr_results.json (task -> category)

Vector PDFs at ICML widths: column 3.25 in, full 6.75 in.
"""
import csv
import glob
import json
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
COL_W, FULL_W = 3.25, 6.75

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
CLOSED_C, OPEN_C = "#2a78d6", "#1baf7a"  # must match COLOR["closed"] / COLOR["open"] in make_figures.py
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdcd8"

CLOSED = {"Codex/gpt-6-astra": "Codex", "Claude Code/Fable 5.1": "Claude Code", "Gemini CLI/3.1 Pro": "Gemini CLI"}
# lanes called out in the text of Section 5
LABEL = {**CLOSED, "GLM-5": "GLM-5", "Kimi K2.5": "Kimi K2.5", "MiniMax M2.5": "MiniMax M2.5",
         "Qwen3-Coder-Next": "Qwen3-Coder-Next", "gpt-oss-120b": "gpt-oss-120b",
         "Qwen3-32B": "Qwen3-32B", "Llama 4 Maverick": "Llama 4 Maverick",
         "DeepSeek V3.2": "DeepSeek V3.2", "Nemotron Nano 3 30B": "Nemotron Nano 3"}


def fig_tools_vs_score():
    rows = list(csv.DictReader(open(ROOT / "bench/paper/table3_efficiency.csv")))
    rows = [r for r in rows if int(r["produced"]) > 0]  # lanes that made at least one video
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    for r in rows:
        x, y = float(r["tool_calls_per_attempt"]), float(r["score"])
        closed = r["model"] in CLOSED
        ax.scatter(x, y, s=16 if closed else 11, color=CLOSED_C if closed else OPEN_C,
                   edgecolor="white", linewidth=0.5, zorder=3)
    # labels: hand-placed offsets so nothing collides
    off = {"Codex": (4, -6), "Claude Code": (0, 6), "Gemini CLI": (4, -6), "GLM-5": (-2, 6), "Kimi K2.5": (-4, -3),
           "MiniMax M2.5": (4, 0), "DeepSeek V3.2": (-4, 2), "Qwen3-Coder-Next": (4, -3), "gpt-oss-120b": (-4, 0),
           "Qwen3-32B": (-2, 7)}
    ha = {"GLM-5": "right", "DeepSeek V3.2": "right", "Claude Code": "center", "Kimi K2.5": "right",
          "gpt-oss-120b": "right"}
    for r in rows:
        name = LABEL.get(r["model"])
        if not name or name not in off:
            continue
        x, y = float(r["tool_calls_per_attempt"]), float(r["score"])
        text = name + (f" ({r['timeouts']} timeouts)" if name == "Qwen3-Coder-Next" else "")
        if name == "Qwen3-32B":  # the two "give up" lanes sit on top of each other at the origin
            ax.annotate("Qwen3-32B,\nLlama 4 Maverick", (x, y), xytext=(1.55, 0.2), textcoords="data", fontsize=5.5,
                        ha="left", va="center", color=INK,
                        arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.4, shrinkA=0, shrinkB=2))
            continue
        ax.annotate(text, (x, y), xytext=off[name], textcoords="offset points", fontsize=5.5,
                    ha=ha.get(name, "left"), va="center", color=INK)
    ax.set_xscale("log")
    ax.set_xlim(1.4, 300)
    ax.set_ylim(-0.03, 1.0)
    ax.set_xticks([2, 5, 10, 20, 50, 100, 200])
    ax.set_xticklabels(["2", "5", "10", "20", "50", "100", "200"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Tool calls per instance (mean, log scale)")
    ax.set_ylabel("VBVR-Pro-Bench score")
    ax.grid(True, color=GRID, linewidth=0.4, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=4, color=CLOSED_C, label="closed agent"),
               Line2D([], [], linestyle="none", marker="o", markersize=3.3, color=OPEN_C, label="OpenCode + open-weight model")]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 0.84), frameon=False, handletextpad=0.2, borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_tools_vs_score.pdf")
    plt.close(fig)


def task_categories():
    f = glob.glob(str(ROOT / "bench/results/codex-gpt-6-astra/*_vbvr_results.json"))[0]
    cat = {}
    for s in json.load(open(f))["samples"]:
        cat[s["task_name"].replace("_data-generator", "")] = s["category"]
    return cat


def fig_heatmap():
    rows = list(csv.DictReader(open(ROOT / "bench/paper/table4_per_task_closed.csv")))
    cat = task_categories()
    agents = list(CLOSED)
    cats = ["Abstraction", "Perception", "Spatiality", "Transformation", "Knowledge"]
    cmap = LinearSegmentedColormap.from_list("blues", ["#f4f7fb", "#c6d9f1", "#7fb0e3", "#2a78d6", "#123f77"])
    fig, axes = plt.subplots(2, 1, figsize=(FULL_W, 2.55), gridspec_kw={"hspace": 1.25})
    for ax, split, title in zip(axes, ["In_Domain", "Out_of_Domain"], ["In-domain (50 tasks)", "Out-of-domain (50 tasks)"]):
        sub = [r for r in rows if r["split"] == split]
        # group by category, then by mean score ascending inside the group
        sub.sort(key=lambda r: (cats.index(cat[r["task"]]), np.mean([float(r[a]) for a in agents])))
        M = np.array([[float(r[a]) for r in sub] for a in agents])
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        ax.set_yticks(range(3))
        ax.set_yticklabels([CLOSED[a] for a in agents])
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels([r["task"].split("_")[0] for r in sub], rotation=90, fontsize=4.2)
        ax.tick_params(axis="x", length=1.5, pad=1)
        ax.tick_params(axis="y", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        # category bands above the panel
        start = 0
        for c in cats:
            n = sum(1 for r in sub if cat[r["task"]] == c)
            if n == 0:
                continue
            ax.plot([start - 0.4, start + n - 0.6], [-0.75, -0.75], color=INK2, linewidth=0.8, clip_on=False)
            ax.text(start + n / 2 - 0.5, -0.95, c, ha="center", va="bottom", fontsize=5.5, color=INK2, clip_on=False)
            if start > 0:
                ax.axvline(start - 0.5, color="white", linewidth=1.2)
            start += n
        ax.set_title(title, loc="left", fontsize=6.5, pad=13, color=INK)
    fig.subplots_adjust(left=0.085, right=0.92, top=0.86, bottom=0.14)
    cax = fig.add_axes([0.935, 0.14, 0.008, 0.72])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("mean score over 5 instances", fontsize=5.5, labelpad=2)
    cb.ax.tick_params(labelsize=5.5, width=0.4, length=2)
    cb.outline.set_linewidth(0.4)
    fig.savefig(OUT / "fig_heatmap.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_tools_vs_score()
    fig_heatmap()
    print("wrote", OUT / "fig_tools_vs_score.pdf", OUT / "fig_heatmap.pdf")
