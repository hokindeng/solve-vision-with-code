#!/usr/bin/env python3
"""Qualitative figures from real bench frames: the comparison wall and the two code+frame listings.

  fig_qualitative.pdf   FULL (6.0 in) x ~3.8 in. Ten tasks in two side-by-side blocks of five rows;
                        columns: first frame | ground truth | CODEX | CLAUDE | GEMINI (last frames), the
                        official evaluator's score of that instance under each agent cell. Rows are grouped
                        by category (label at the left of the block); the task id is written under each row.
  fig_code_O56.pdf      FULL x ~2.9 in. The trimmed Codex solve.py for O-56 (paper/figures/qualitative/
  fig_code_G13.pdf      O-56_raven.py, G-13_grid_number_sequence.py) with light line numbers, beside the
                        task's first frame and the Codex last frame (score in its label, so nothing
                        covers the answer cell).

Inputs (read-only):
  ~/Workspace/svc-media/index.json            frames and per-instance scores
  bench/paper/table4_per_task_closed.csv      per-task means, used by --check against index.json
Style: paper/figures/STYLE.md. Fonts embedded (pdf.fonttype 42); only the frames are raster.

    python3 paper/figures/make_qualitative.py            # writes the three PDFs
    python3 paper/figures/make_qualitative.py --check    # verifies scores and listings, exit 1 on mismatch
"""
import csv
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

INK, GREY_1, GREY_2, GREY_3 = "#111111", "#555555", "#999999", "#dddddd"
AGENT, AGENT_OPEN, VIDEO = "#2a5db0", "#a9c0e6", "#a8a29a"
FULL, WIDE, HALF, WRAP = 6.0, 5.4, 2.9, 2.7          # inches, = 1.0 / 0.9 / 0.48 / 0.45 \textwidth

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
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
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})
MONO = "DejaVu Sans Mono"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
INDEX = Path.home() / "Workspace" / "svc-media" / "index.json"
TABLE4 = ROOT / "bench" / "paper" / "table4_per_task_closed.csv"
LISTINGS = HERE / "qualitative"

LANES = ["codex", "claude", "gemini"]
COLS = ["first frame", "ground truth", "CODEX", "CLAUDE", "GEMINI"]

# Ten tasks, instance 00000 each, grouped by category, in two side-by-side blocks of five rows. Chosen so
# that every category appears, the frames are visually distinct, and the wall carries the failures the
# text discusses: G-43 (Codex and Claude 0.0), G-218 (Codex and Claude 0.0, Gemini 1.0), O-32 (Codex and
# Gemini fail), O-52 and O-56 (partial scores). Scores are printed under the thumbnails, not on them,
# because on a 0.5 in thumbnail a badge would cover the answer cell of O-56 and the ball of O-32.
WALL = [
    [("Abstraction", ["O-56_raven", "O-47_sliding_puzzle", "O-22_construction_stack"]),
     ("Knowledge", ["O-52_traffic_light", "G-43_understand_scene_structure"])],
    [("Perception", ["G-174_arrange_circles_by_circumference", "G-218_identify_largest_angle_in_triangle"]),
     ("Spatiality", ["G-13_grid_number_sequence"]),
     ("Transformation", ["O-64_animal_matching", "O-32_rolling_ball"])],
]
IDX = 0

CODE_FIGS = [
    dict(short="O-56_raven", listing="O-56_raven.py", out="fig_code_O56.pdf"),
    dict(short="G-13_grid_number_sequence", listing="G-13_grid_number_sequence.py", out="fig_code_G13.pdf"),
]


# ---- data -------------------------------------------------------------------------------------

def load_index():
    d = json.load(open(INDEX))
    return {t["short"]: t for t in d["tasks"].values()}


def load_table4():
    rows = list(csv.DictReader(open(TABLE4)))
    cols = {"codex": "Codex/gpt-6-astra", "claude": "Claude Code/Fable 5.1", "gemini": "Gemini CLI/3.1 Pro"}
    return {r["task"]: {l: float(r[c]) for l, c in cols.items()} for r in rows}


def label_of(short):
    tid, _, name = short.partition("_")
    return f"{tid} {name.replace('_', ' ')}"


# ---- drawing helpers ------------------------------------------------------------------------------

def thumb(fig, rect, path, px=300):
    ax = fig.add_axes(rect)
    im = Image.open(path).convert("RGB")
    if im.size != (px, px):
        im = im.resize((px, px), Image.LANCZOS)
    ax.imshow(im, interpolation="lanczos")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_edgecolor(GREY_3); s.set_linewidth(0.5)
    return ax


# ---- fig_qualitative ------------------------------------------------------------------------------

def draw_wall(by_short):
    W = FULL
    T = 0.50                     # thumbnail side, inches
    gap = 0.05                   # between thumbnails
    cat_w = 0.16                 # rotated category label column
    text_h = 0.20                # two 5.5 pt lines under each row: task label left, scores under the agents
    row_h = T + text_h + 0.04
    header_h = 0.14
    n_rows = max(sum(len(ts) for _, ts in block) for block in WALL)
    block_w = cat_w + 5 * T + 4 * gap
    block_gap = W - 2 * block_w
    H = header_h + n_rows * row_h - 0.04
    fig = plt.figure(figsize=(W, H))
    fx, fy = 1.0 / W, 1.0 / H

    def rect(x, top, w, h):
        return [x * fx, 1 - (top + h) * fy, w * fx, h * fy]

    for b, block in enumerate(WALL):
        x0 = b * (block_w + block_gap)
        xs = [x0 + cat_w + c * (T + gap) for c in range(5)]
        for c, name in enumerate(COLS):
            fig.text((xs[c] + T / 2) * fx, 1 - (header_h - 0.035) * fy, name, fontsize=6.5, color=GREY_1,
                     ha="center", va="bottom")
        r = 0
        for cat, shorts in block:
            top_cat = header_h + r * row_h
            n = len(shorts)
            span = n * row_h - 0.04
            fig.text((x0 + 0.05) * fx, 1 - (top_cat + span / 2) * fy, cat, fontsize=6.5, color=GREY_1,
                     rotation=90, ha="center", va="center")
            fig.add_artist(plt.Line2D([(x0 + 0.12) * fx] * 2,
                                      [1 - top_cat * fy, 1 - (top_cat + span) * fy],
                                      transform=fig.transFigure, color=GREY_3, lw=0.5))
            for s in shorts:
                t = by_short[s]
                inst = t["instances"][IDX]
                top = header_h + r * row_h
                thumb(fig, rect(xs[0], top, T, T), inst["gt"]["first_frame"])
                thumb(fig, rect(xs[1], top, T, T), inst["gt"]["last_frame_from_video"])
                ty = 1 - (top + T + 0.03) * fy
                for c, lane in enumerate(LANES):
                    ln = inst["lanes"][lane]
                    thumb(fig, rect(xs[2 + c], top, T, T), ln["last_frame"])
                    fig.text((xs[2 + c] + T / 2) * fx, ty, f"{ln['score']:.3f}", fontsize=6, color=INK,
                             ha="center", va="top")
                split = "in-domain" if t["split"].startswith("In") else "out-of-domain"
                label = "\n".join(textwrap.wrap(f"{label_of(s)} \u00b7 {split}", 30, break_on_hyphens=False)[:2])
                fig.text(xs[0] * fx, ty, label, fontsize=5.5, color=INK, ha="left", va="top", linespacing=1.15)
                r += 1
    return fig


# ---- fig_code_* -----------------------------------------------------------------------------------

def draw_code(by_short, spec):
    t = by_short[spec["short"]]
    inst = t["instances"][IDX]
    lane = inst["lanes"]["codex"]
    lines = (LISTINGS / spec["listing"]).read_text().rstrip("\n").split("\n")
    W = FULL
    T = 1.25                                 # frame side
    lab = 0.12                               # label line above each frame
    H = 2 * (lab + T) + 0.10
    fig = plt.figure(figsize=(W, H))
    fx, fy = 1.0 / W, 1.0 / H

    def rect(x, top, w, h):
        return [x * fx, 1 - (top + h) * fy, w * fx, h * fy]

    code_w = W - T - 0.18
    cax = fig.add_axes(rect(0, 0, code_w, H))
    cax.set_xlim(0, 1); cax.set_ylim(0, 1); cax.set_xticks([]); cax.set_yticks([])
    for s in cax.spines.values():
        s.set_visible(True); s.set_edgecolor(GREY_3); s.set_linewidth(0.5)
    n = len(lines)
    pad = 0.045
    step = (1 - 2 * pad) / max(n - 1, 1)
    gutter_in = 0.28
    for k, ln in enumerate(lines):
        y = 1 - pad - k * step
        cax.text(0.020, y, f"{k + 1:>2}", transform=cax.transAxes, fontsize=5.5, family=MONO,
                 color=GREY_2, ha="left", va="center")
        cax.text(gutter_in / code_w, y, ln, transform=cax.transAxes, fontsize=7.2, family=MONO,
                 color=INK, ha="left", va="center")
    fig.text(0.01 * fx + 0.0, 1 - (H + 0.04) * fy, f"solve.py written by Codex for {label_of(spec['short'])}, instance {IDX:05d}",
             fontsize=6.5, color=GREY_1, ha="left", va="top")

    x_img = W - T
    fig.text((x_img + T / 2) * fx, 1 - (lab - 0.03) * fy, "first frame", fontsize=6.5, color=GREY_1,
             ha="center", va="bottom")
    thumb(fig, rect(x_img, lab, T, T), inst["gt"]["first_frame"], px=500)
    top2 = lab + T + 0.10 + lab
    fig.text((x_img + T / 2) * fx, 1 - (top2 - 0.03) * fy, f"Codex last frame \u00b7 score {lane['score']:.3f}",
             fontsize=6.5, color=GREY_1, ha="center", va="bottom")
    thumb(fig, rect(x_img, top2, T, T), lane["last_frame"], px=500)
    return fig


# ---- check ----------------------------------------------------------------------------------------

def check(by_short, t4):
    ok = True
    print("wall cells (instance %05d), scores from index.json; task means vs table4:" % IDX)
    for block in WALL:
        for cat, shorts in block:
            for s in shorts:
                t = by_short[s]
                if t["category"] != cat:
                    print(f"  {s}: category {t['category']} != group {cat}"); ok = False
                inst = t["instances"][IDX]
                cell = " ".join(f"{l}={inst['lanes'][l]['score']:.3f}" for l in LANES)
                means = {l: sum(i["lanes"][l]["score"] for i in t["instances"]) / len(t["instances"]) for l in LANES}
                ref = t4[s]
                for l in LANES:
                    if abs(means[l] - ref[l]) > 1e-6:
                        print(f"  {s}: mean {l} {means[l]:.6f} != table4 {ref[l]:.6f}"); ok = False
                print(f"  {cat:<15} {s:<42} {cell}   means " + " ".join(f"{means[l]:.3f}" for l in LANES))
    for spec in CODE_FIGS:
        lines = (LISTINGS / spec["listing"]).read_text().rstrip("\n").split("\n")
        s = by_short[spec["short"]]["instances"][IDX]["lanes"]["codex"]["score"]
        print(f"  {spec['listing']}: {len(lines)} lines, longest {max(map(len, lines))} cols, Codex score {s:.3f}")
        ok &= len(lines) <= 25 and max(map(len, lines)) <= 56 and s >= 1.0
    print("check", "ok" if ok else "FAILED")
    return ok


def main(argv):
    by_short = load_index()
    t4 = load_table4()
    if "--check" in argv:
        return 0 if check(by_short, t4) else 1
    fig = draw_wall(by_short)
    fig.savefig(HERE / "fig_qualitative.pdf", format="pdf")
    print("wrote", HERE / "fig_qualitative.pdf")
    for spec in CODE_FIGS:
        fig = draw_code(by_short, spec)
        fig.savefig(HERE / spec["out"], format="pdf")
        print("wrote", HERE / spec["out"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
