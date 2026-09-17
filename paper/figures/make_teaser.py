#!/usr/bin/env python3
"""Teaser figure (fig_teaser.pdf): one row, three panels, 6.75 in wide, vector PDF.

  left    the first frame of a real bench instance + its prompt (truncated)
  middle  "coding agent writes solve.py" arrow with real lines from a Codex run
  right   best coding agent vs best video model, with ID -> OOD arrows

Inputs (all read-only, resolved relative to the repo root):
  runs/gpt-6-astra/G-35_hit_target_after_bounce/app/{first_frame.png, prompt.txt, solve.py, output/final_frame.png}
    — a real Codex (gpt-6-astra) run on the G-35 generator (the bench data directory
      project/vbvr-pro-bench-data is not present on this machine, so the repo-local run is used).
  Numbers are from bench/paper/table1_leaderboard.csv:
    Codex x gpt-6-astra           0.9228 overall, 0.879 ID -> 0.9671 OOD
    VBVR-Pro-Wan2.2-I2V-A14B (RL) 0.670  overall, 0.808  ID -> 0.532  OOD

Only the frame rasters are embedded; everything else stays vector.
    python3 paper/figures/make_teaser.py
"""
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RUN = ROOT / "runs" / "gpt-6-astra" / "G-35_hit_target_after_bounce" / "app"
OUT_PDF = HERE / "fig_teaser.pdf"
OUT_PNG = HERE / "fig_teaser.png"

# ---- numbers (bench/paper/table1_leaderboard.csv) ---------------------------------
AGENT = dict(name="Codex $\\times$ gpt-6-astra", overall=0.923, id=0.879, ood=0.967)
VIDEO = dict(name="VBVR-Pro-Wan2.2-I2V-A14B\n(RL-trained video model)", overall=0.670, id=0.808, ood=0.532)

# ---- palette ------------------------------------------------------------------
C_AGENT = "#4C7DBF"   # resourceblue from main.tex
C_VIDEO = "#B0B0B0"
C_INK = "#222222"
C_MUTED = "#666666"
C_CODEBG = "#F5F5F5"
C_UP = "#2E8B57"
C_DOWN = "#C0392B"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 7.5,
    "pdf.fonttype": 42,
    "axes.linewidth": 0.6,
})


def load_code_lines(path: Path, n: int = 6):
    """Pick n representative lines from the real solve.py: the ones that carry the reasoning."""
    lines = path.read_text().splitlines()
    wanted = [
        "gold = (a[:,:,0] > 100)",
        "n, labels, stats, centroids = cv2.connectedComponentsWithStats",
        "tx,ty = min(targets,key=lambda p:p[1])",
        "# Unfold the reflected path",
        "x = fold(x0+(end_x-x0)*t,lo_x,hi_x)",
        "draw.ellipse((cx-radius,cy-radius,cx+radius,cy+radius),fill=color)",
    ]
    picked = []
    for w in wanted:
        for ln in lines:
            if ln.strip().startswith(w) or ln.strip() == w:
                picked.append(ln.strip())
                break
    picked = picked[:n]
    # shorten very long lines so the panel stays readable at 7 pt
    short = []
    for ln in picked:
        short.append(ln if len(ln) <= 44 else ln[:41] + "...")
    return short


def main():
    first = Image.open(RUN / "first_frame.png").convert("RGB")
    final = Image.open(RUN / "output" / "final_frame.png").convert("RGB")
    prompt = (RUN / "prompt.txt").read_text().strip()
    code = load_code_lines(RUN / "solve.py")

    # downsample the rasters: they are 1024^2, the panel is ~1.3 in -> 300 dpi needs ~400 px
    first = first.resize((400, 400), Image.LANCZOS)
    final = final.resize((400, 400), Image.LANCZOS)

    fig = plt.figure(figsize=(6.75, 2.15))
    # panel geometry in figure fractions
    ax_frame = fig.add_axes([0.005, 0.30, 0.185, 0.66])
    ax_final = fig.add_axes([0.565, 0.30, 0.185, 0.66])
    ax_code = fig.add_axes([0.20, 0.05, 0.355, 0.92]); ax_code.axis("off")
    ax_bar = fig.add_axes([0.815, 0.24, 0.18, 0.62])

    # ---- left: first frame + prompt ------------------------------------------------
    ax_frame.imshow(first, interpolation="lanczos")
    ax_frame.set_xticks([]); ax_frame.set_yticks([])
    for s in ax_frame.spines.values():
        s.set_edgecolor(C_MUTED); s.set_linewidth(0.6)
    ax_frame.set_title("first frame + prompt", fontsize=7.5, color=C_INK, pad=3)
    ptxt = textwrap.shorten(prompt, width=150, placeholder=" ...")
    ptxt = "\n".join(textwrap.wrap(ptxt, width=42))
    fig.text(0.005, 0.245, ptxt, fontsize=5.4, color=C_MUTED, va="top", ha="left",
             family="serif", linespacing=1.25)

    # ---- middle: arrow + code ------------------------------------------------------
    ax_code.set_xlim(0, 1); ax_code.set_ylim(0, 1)
    ax_code.add_patch(FancyArrowPatch((0.0, 0.74), (1.0, 0.74), arrowstyle="-|>",
                                      mutation_scale=9, lw=1.0, color=C_AGENT,
                                      transform=ax_code.transAxes, zorder=1))
    ax_code.text(0.5, 0.93, "coding agent writes solve.py", ha="center", va="center",
                 fontsize=8, color=C_AGENT, fontweight="bold")
    ax_code.text(0.5, 0.845, "sees only the frame and the prompt; renders the answer video with code",
                 ha="center", va="center", fontsize=5.6, color=C_MUTED)
    box = FancyBboxPatch((0.03, 0.06), 0.94, 0.50, boxstyle="round,pad=0.01,rounding_size=0.015",
                         fc=C_CODEBG, ec=C_MUTED, lw=0.5, transform=ax_code.transAxes, zorder=2)
    ax_code.add_patch(box)
    y = 0.50
    for ln in code:
        ax_code.text(0.06, y, ln, fontsize=5.3, family="monospace", color=C_INK,
                     va="center", ha="left", zorder=3)
        y -= 0.078
    ax_code.text(0.06, 0.585, "solve.py  (Codex, G-35 hit-target-after-bounce, 67 lines)", fontsize=5.2,
                 color=C_MUTED, va="bottom", ha="left", style="italic")

    # ---- final frame -------------------------------------------------------------
    ax_final.imshow(final, interpolation="lanczos")
    ax_final.set_xticks([]); ax_final.set_yticks([])
    for s in ax_final.spines.values():
        s.set_edgecolor(C_AGENT); s.set_linewidth(0.8)
    ax_final.set_title("rendered last frame", fontsize=7.5, color=C_INK, pad=3)
    fig.text(0.6575, 0.245, "scored by the official\nrule-based evaluator", fontsize=5.6,
             color=C_MUTED, va="top", ha="center", linespacing=1.25)

    # ---- right: bars + ID/OOD arrows ------------------------------------------------
    names = ["best coding\nagent", "best video\nmodel"]
    vals = [AGENT["overall"], VIDEO["overall"]]
    cols = [C_AGENT, C_VIDEO]
    x = [0, 1]
    ax_bar.bar(x, vals, width=0.62, color=cols, edgecolor="none", zorder=2)
    for xi, v in zip(x, vals):
        ax_bar.text(xi, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=7.2,
                    fontweight="bold", color=C_INK)
    ax_bar.set_xticks(x); ax_bar.set_xticklabels(names, fontsize=6.4)
    ax_bar.set_ylim(0, 1.13); ax_bar.set_xlim(-0.45, 1.62); ax_bar.set_yticks([0, 0.5, 1.0])
    ax_bar.tick_params(axis="y", labelsize=6, length=2, width=0.5)
    ax_bar.tick_params(axis="x", length=0)
    for s in ["top", "right"]:
        ax_bar.spines[s].set_visible(False)
    ax_bar.spines["left"].set_linewidth(0.5); ax_bar.spines["bottom"].set_linewidth(0.5)
    ax_bar.set_ylabel("score", fontsize=6.4, labelpad=1)
    ax_bar.set_title("500 instances, ID + OOD", fontsize=7.5, color=C_INK, pad=3)

    # ID -> OOD annotations inside the bars
    def arrow_note(xi, lo, hi, up, color):
        txt = f"ID {lo:.3f} $\\rightarrow$ OOD {hi:.3f}"
        ax_bar.text(xi, 0.06, txt, ha="center", va="bottom", fontsize=4.9, color="white" if xi == 0 else C_INK,
                    rotation=90)
        sym = "$\\uparrow$" if up else "$\\downarrow$"
        ax_bar.text(xi + 0.33, 0.02, sym, ha="left", va="bottom", fontsize=9, color=color)

    arrow_note(0, AGENT["id"], AGENT["ood"], True, C_UP)
    arrow_note(1, VIDEO["id"], VIDEO["ood"], False, C_DOWN)
    fig.text(0.905, 0.02, "OOD $-$ ID: +0.089 vs $-$0.276", fontsize=5.6, ha="center",
             va="bottom", color=C_MUTED)

    fig.savefig(OUT_PDF, format="pdf")
    fig.savefig(OUT_PNG, format="png", dpi=300)
    print(f"wrote {OUT_PDF} and {OUT_PNG}")


if __name__ == "__main__":
    main()
