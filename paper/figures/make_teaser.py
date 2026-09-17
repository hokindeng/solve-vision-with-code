#!/usr/bin/env python3
"""Teaser (fig_teaser.pdf + .png): real bench frames, real solve.py lines, the leaderboard.

Layout, WIDE = 5.4 in (0.9 \\textwidth in the pr template), 2.56 in tall:

  (a) three benchmark instances, one per row, each from a different category:
        first frame  ->  seven lines of the Codex solve.py that computed the answer  ->  Codex last frame
        (score badge) beside the ground-truth last frame; the prompt, truncated, under the first frame.
  (b) top-3 coding agents vs top-3 video models from table1 (horizontal bars, 3 decimals), the best
        video model's score as a dashed reference line, and the in-domain -> out-of-domain shift of
        the best agent and the RL-trained video model.

Inputs (read-only):
  ~/Workspace/svc-media/index.json         frames, prompts, solve.py paths, per-instance scores
  bench/paper/table1_leaderboard.csv       every number drawn in panel (b)
Style: paper/figures/STYLE.md.  Fonts are embedded (pdf.fonttype 42); only the six frames are raster.

    python3 paper/figures/make_teaser.py            # writes fig_teaser.pdf and fig_teaser.png
    python3 paper/figures/make_teaser.py --check    # re-reads the CSV / index and verifies what is drawn
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
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
TABLE1 = ROOT / "bench" / "paper" / "table1_leaderboard.csv"
OUT_PDF, OUT_PNG = HERE / "fig_teaser.pdf", HERE / "fig_teaser.png"

# ---- the three instances ----------------------------------------------------------------------
# Chosen from index.json (see list_candidates): Codex score 1.0 at instance 00000, three categories,
# visually distinct first frames. The code lines are verbatim from the agent's solve.py (matched by
# prefix, shortened only by dropping characters past the column limit, which is marked with an
# ellipsis); they are the lines that compute the answer, not the I/O scaffolding.
TEASER = [
    dict(short="O-39_maze", lane="codex", idx=0, lines=[
        "queue = deque([start])",
        "while queue:",
        "    r, c = queue.popleft()",
        "    if (r, c) == end:",
        "        break",
        "    for p in ((r-1,c), (r+1,c), (r,c-1), (r,c+1)):",
    ]),
    dict(short="O-56_raven", lane="codex", idx=0, lines=[
        "# Each row shifts the three symbols one position left",
        "source = base[420:590, 85:255]",
        "sy, sx = np.where(mask)",
        "ty, tx = sy + 761, sx + 767",
        "colors = source[sy, sx]",
        "frame[ty[visible],tx[visible]] = colors[visible]",
    ]),
    dict(short="O-62_gravity_physics", lane="codex", idx=0, lines=[
        "g, restitution = 6.9, .70",
        "duration = (v + math.sqrt(v*v + 2*g*h))/g",
        "impact = v - g*duration",
        "v = -restitution*impact",
        "return max(0., h0+v0*dt-.5*g*dt*dt), v0-g*dt",
    ]),
]
CODE_COLS = 38          # characters per code line at 4.8 pt DejaVu Sans Mono in a 1.62 in box
CODE_ROWS = 7           # display lines per code panel (a wrapped statement uses two)
PROMPT_CHARS = 58

# ---- data ---------------------------------------------------------------------------------------

def load_index():
    d = json.load(open(INDEX))
    by_short = {t["short"]: t for t in d["tasks"].values()}
    return by_short


def load_table1():
    rows = list(csv.DictReader(open(TABLE1)))
    for r in rows:
        for k in ("overall", "in_domain", "out_of_domain"):
            r[k] = float(r[k])
        r["produced"] = int(r["produced"])
    return rows


def pick_bars(rows, n=3):
    agents = [r for r in rows if r["kind"] == "coding agent"]
    video = [r for r in rows if r["kind"] == "video model"]
    agents.sort(key=lambda r: -r["overall"]); video.sort(key=lambda r: -r["overall"])
    return agents[:n], video[:n]


def list_candidates(by_short):
    """Helper used to choose the teaser rows: every task where Codex scores 1.0 at instance 0."""
    for s, t in sorted(by_short.items(), key=lambda kv: (kv[1]["category"], kv[0])):
        i0 = t["instances"][0]
        if i0["lanes"]["codex"]["score"] >= 1.0:
            print(f"{t['category']:<15} {t['split']:<17} {s}")


def real_lines(solve_path, wanted, cols=CODE_COLS):
    """Return the requested lines exactly as they appear in solve.py, each verified to exist.
    Indentation is the spec's (the first level of the agent's block is dropped). A statement longer
    than the panel is wrapped onto a continuation line with a hanging indent, as the qualitative
    listings do; a comment longer than the panel is cut with an ellipsis."""
    src = [ln.rstrip() for ln in Path(solve_path).read_text().splitlines()]
    out = []
    for w in wanted:
        key = w.strip()
        hit = next((ln for ln in src if ln.strip().startswith(key)), None)
        if hit is None:
            raise SystemExit(f"line not found in {solve_path}: {w!r}")
        indent = " " * (len(w) - len(w.lstrip()))
        text = indent + hit.strip()
        if len(text) <= cols:
            out.append(text)
        elif text.lstrip().startswith("#"):
            out.append(text[: cols - 1] + "\u2026")
        else:
            cut = text.rfind(" ", 0, cols)
            out.append(text[:cut])
            out.append(indent + "    " + text[cut + 1:])
    if len(out) > CODE_ROWS:
        raise SystemExit(f"{solve_path}: {len(out)} display lines, panel holds {CODE_ROWS}")
    return out


def truncate(s, n):
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def thumb(fig, rect, path, edge=GREY_3, lw=0.5, px=360):
    ax = fig.add_axes(rect)
    im = Image.open(path).convert("RGB")
    if im.size != (px, px):
        im = im.resize((px, px), Image.LANCZOS)
    ax.imshow(im, interpolation="lanczos")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_edgecolor(edge); s.set_linewidth(lw)
    return ax


def badge(ax, score, fs=5.5):
    ax.text(0.96, 0.05, f"{score:.3f}", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=fs, color=INK, zorder=5,
            bbox=dict(boxstyle="square,pad=0.22", fc="white", ec=GREY_3, lw=0.5, alpha=0.92))


def arrow(fig, x0, x1, y):
    fig.add_artist(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=5, lw=0.6, color=GREY_1, shrinkA=0, shrinkB=0))


# ---- drawing --------------------------------------------------------------------------------------

def draw(by_short, rows):
    W, H = WIDE, 2.56
    fig = plt.figure(figsize=(W, H))
    fx, fy = 1.0 / W, 1.0 / H                         # inches -> figure fraction

    # panel (a) geometry, inches from the left / from the top
    T = 0.56                                          # thumbnail side
    x_first = 0.0
    x_code = x_first + T + 0.10
    code_w = 1.62
    x_last = x_code + code_w + 0.10
    x_gt = x_last + T + 0.05
    header_h = 0.26
    row_h = 0.72                                      # thumb + one text line + gap
    top0 = 0.02 + header_h

    def rect(x, top, w, h):                           # inches -> [l, b, w, h] in figure fraction
        return [x * fx, 1 - (top + h) * fy, w * fx, h * fy]

    # panel letters and column headers
    fig.text(0.0, 1 - 0.02 * fy, "(a)", fontsize=8, fontweight="bold", color=INK, ha="left", va="top")
    hy = 1 - (top0 - 0.04) * fy
    fig.text(x_first * fx, hy, "first frame + prompt", fontsize=6.5, color=GREY_1, ha="left", va="bottom")
    fig.text((x_code + code_w / 2) * fx, hy, "Codex writes solve.py", fontsize=6.5, color=GREY_1, ha="center", va="bottom")
    fig.text((x_last + T / 2) * fx, hy, "Codex last frame", fontsize=6.5, color=GREY_1, ha="center", va="bottom")
    fig.text((x_gt + T / 2) * fx, hy, "ground truth", fontsize=6.5, color=GREY_1, ha="center", va="bottom")

    for r, spec in enumerate(TEASER):
        t = by_short[spec["short"]]
        inst = t["instances"][spec["idx"]]
        lane = inst["lanes"][spec["lane"]]
        top = top0 + r * row_h
        ax1 = thumb(fig, rect(x_first, top, T, T), inst["gt"]["first_frame"])
        # code panel: real lines, GREY_3 frame, no fill
        cax = fig.add_axes(rect(x_code, top, code_w, T))
        cax.set_xlim(0, 1); cax.set_ylim(0, 1); cax.set_xticks([]); cax.set_yticks([])
        for s in cax.spines.values():
            s.set_visible(True); s.set_edgecolor(GREY_3); s.set_linewidth(0.5)
        lines = real_lines(lane["solve"], spec["lines"])
        for k, ln in enumerate(lines):
            cax.text(0.03, 1 - (k + 0.8) / (CODE_ROWS + 0.5), ln, transform=cax.transAxes, fontsize=4.8,
                     family=MONO, color=INK, ha="left", va="center")
        ax2 = thumb(fig, rect(x_last, top, T, T), lane["last_frame"])
        badge(ax2, lane["score"])
        thumb(fig, rect(x_gt, top, T, T), inst["gt"]["last_frame_from_video"])
        ymid = 1 - (top + T / 2) * fy
        arrow(fig, (x_first + T + 0.015) * fx, (x_code - 0.015) * fx, ymid)
        arrow(fig, (x_code + code_w + 0.015) * fx, (x_last - 0.015) * fx, ymid)
        # prompt under the first frame, task label under the two result frames
        ty = 1 - (top + T + 0.045) * fy
        fig.text(x_first * fx, ty, truncate(inst["prompt"], PROMPT_CHARS), fontsize=5.5, color=INK,
                 ha="left", va="top")
        tid, _, tname = spec["short"].partition("_")
        split = "in-domain" if t["split"].startswith("In") else "out-of-domain"
        fig.text((x_gt + T) * fx, ty, f"{tid} {tname.replace('_', ' ')} · {t['category']} · {split}",
                 fontsize=5.5, color=GREY_1, ha="right", va="top")

    # panel (b): bars
    agents, video = pick_bars(rows)
    bx = x_gt + T + 0.30
    bw = W - bx
    fig.text(bx * fx, 1 - 0.02 * fy, "(b)", fontsize=8, fontweight="bold", color=INK, ha="left", va="top")
    fig.text((bx + 0.22) * fx, 1 - 0.03 * fy, "coding agents", fontsize=6.5, color=AGENT, ha="left", va="top")
    fig.text((bx + 0.22 + 0.62) * fx, 1 - 0.03 * fy, "video models", fontsize=6.5, color=VIDEO, ha="left", va="top")
    bars_top = top0
    slot = 0.285
    bar_h = 0.10
    bax = fig.add_axes(rect(bx, bars_top, bw, slot * 6))
    bax.set_xlim(0, 1.2); bax.set_ylim(6 * slot, 0)   # values sit inside the axes
    bax.axis("off")
    items = [(r, AGENT) for r in agents] + [(r, VIDEO) for r in video]
    best_video = video[0]["overall"]
    for k, (r, col) in enumerate(items):
        y_label = k * slot + 0.02
        y_bar = k * slot + 0.13
        name = r["model"].replace(" (RL-trained video model)", " (RL)")
        bax.text(0, y_label, name, fontsize=6.5, color=GREY_1 if col == VIDEO else INK, ha="left", va="top",
                 zorder=4, bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none"))
        bax.add_patch(Rectangle((0, y_bar), r["overall"], bar_h, fc=col, ec="none"))
        bax.text(r["overall"] + 0.015, y_bar + bar_h / 2, f"{r['overall']:.3f}", fontsize=7, color=INK,
                 ha="left", va="center")
    # the best video model's score, dropped through the agent bars onto its own bar
    bax.plot([best_video, best_video], [0.13, 3 * slot + 0.13], color=GREY_2, lw=0.6, ls=(0, (3, 2)), zorder=3)

    # in-domain -> out-of-domain shift: best agent vs the trained video model
    a, v = agents[0], video[0]
    ny = 1 - (bars_top + slot * 6 + 0.06) * fy
    fig.text(bx * fx, ny, "in-domain → out-of-domain", fontsize=6.5, color=GREY_1, ha="left", va="top")
    fig.text(bx * fx, ny - 0.115 * fy,
             f"{a['model']}: {a['in_domain']:.3f} → {a['out_of_domain']:.3f}",
             fontsize=6.5, color=INK, ha="left", va="top")
    fig.text(bx * fx, ny - 0.225 * fy,
             f"RL-trained video model: {v['in_domain']:.3f} → {v['out_of_domain']:.3f}",
             fontsize=6.5, color=GREY_1, ha="left", va="top")
    return fig


def check(by_short, rows):
    ok = True
    agents, video = pick_bars(rows)
    print("panel (b) bars, from", TABLE1.relative_to(ROOT))
    for r in agents + video:
        print(f"  {r['kind']:<12} {r['model']:<52} {r['overall']:.3f}  ID {r['in_domain']:.3f}  OOD {r['out_of_domain']:.3f}")
    a, v = agents[0], video[0]
    ok &= a["out_of_domain"] > a["in_domain"] and v["out_of_domain"] < v["in_domain"]
    print("panel (a) rows, from index.json")
    for spec in TEASER:
        inst = by_short[spec["short"]]["instances"][spec["idx"]]
        s = inst["lanes"][spec["lane"]]["score"]
        print(f"  {spec['short']:<24} {spec['lane']} idx {spec['idx']} score {s:.3f}")
        ok &= s >= 1.0
        real_lines(inst["lanes"][spec["lane"]]["solve"], spec["lines"])   # raises if a line is not real
    cats = {by_short[s["short"]]["category"] for s in TEASER}
    ok &= len(cats) == len(TEASER)
    print("check", "ok" if ok else "FAILED")
    return ok


def main(argv):
    by_short = load_index()
    rows = load_table1()
    if "--list" in argv:
        list_candidates(by_short); return 0
    if "--check" in argv:
        return 0 if check(by_short, rows) else 1
    fig = draw(by_short, rows)
    fig.savefig(OUT_PDF, format="pdf")
    fig.savefig(OUT_PNG, format="png", dpi=300)
    print(f"wrote {OUT_PDF} and {OUT_PNG}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
