#!/usr/bin/env python3
"""Solve the 3x3 progressive-matrix puzzle and render the answer as a video.

Rule: every row (and column) contains each of the three shapes exactly once
(cross, pentagon, square). Row 3 has square, cross -> missing shape is the
orange pentagon. We copy the pentagon's exact pixels from a visible cell and
trace it progressively into the bottom-right cell after fading out the "?".
"""
import os
import subprocess
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 35, 16

base = np.array(Image.open(SRC).convert("RGB"))
H, W, _ = base.shape

# --- grid geometry (3x3, thick black lines) -------------------------------
dark = base.sum(axis=2) < 100
line_rows = np.where(dark.mean(axis=1) > 0.9)[0]
line_cols = np.where(dark.mean(axis=0) > 0.9)[0]

def cell_bounds(lines, size):
    """Return list of (start, end) for the 3 cells separated by line pixels."""
    inside = np.ones(size, bool)
    inside[lines] = False
    bounds, start = [], None
    for i in range(size):
        if inside[i] and start is None:
            start = i
        if (not inside[i] or i == size - 1) and start is not None:
            end = i if inside[i] else i - 1
            if end - start > 20:
                bounds.append((start, end))
            start = None
    return bounds

rows = cell_bounds(line_rows, H)
cols = cell_bounds(line_cols, W)
assert len(rows) == 3 and len(cols) == 3, (rows, cols)

def cell(r, c):
    (y0, y1), (x0, x1) = rows[r], cols[c]
    return y0, y1 + 1, x0, x1 + 1

# --- identify shapes per cell by colour -------------------------------------
def shape_of(r, c):
    y0, y1, x0, x1 = cell(r, c)
    reg = base[y0 + 3:y1 - 3, x0 + 3:x1 - 3].astype(int)
    m = reg.sum(axis=2) < 740
    if m.sum() == 0:
        return None
    col = tuple(np.median(reg[m], axis=0).astype(int))
    if abs(col[0] - col[1]) < 10 and abs(col[1] - col[2]) < 10:
        return "question"
    return col  # colour identifies the shape (all shapes single-colour)

grid = [[shape_of(r, c) for c in range(3)] for r in range(3)]
visible = {s for row in grid for s in row if s not in (None, "question")}
row2 = {grid[2][0], grid[2][1]}
answer = (visible - row2).pop()          # the colour of the missing shape
# pick a reference cell holding the answer shape
ref = next((r, c) for r in range(3) for c in range(3) if grid[r][c] == answer)

# --- extract the reference shape mask, relative to its cell centre -----------
ry0, ry1, rx0, rx1 = cell(*ref)
reg = base[ry0:ry1, rx0:rx1].astype(int)
mask = np.all(np.abs(reg - np.array(answer)) < 40, axis=2)
ys, xs = np.where(mask)
ref_cy, ref_cx = (ry0 + ry1 - 1) / 2.0, (rx0 + rx1 - 1) / 2.0
ty0, ty1, tx0, tx1 = cell(2, 2)
tgt_cy, tgt_cx = (ty0 + ty1 - 1) / 2.0, (tx0 + tx1 - 1) / 2.0
dy, dx = int(round(tgt_cy - ref_cy)), int(round(tgt_cx - ref_cx))
tys, txs = ys + ry0 + dy, xs + rx0 + dx
colour = base[ys + ry0, xs + rx0]        # exact colours of the reference pixels

# --- perimeter ordering: angle around the shape centre (clockwise from top) ---
scy, scx = (ys.min() + ys.max()) / 2.0, (xs.min() + xs.max()) / 2.0
ang = np.arctan2(xs - scx, -(ys - scy))  # 0 at top, increasing clockwise
ang = np.mod(ang, 2 * np.pi)
order_t = ang / (2 * np.pi)

# --- "?" pixels in the target cell -----------------------------------------
treg = base[ty0:ty1, tx0:tx1].astype(int)
qmask = treg.sum(axis=2) < 740
qys, qxs = np.where(qmask)
qys, qxs = qys + ty0, qxs + tx0
qcol = base[qys, qxs].astype(float)

# --- render frames -----------------------------------------------------------
FADE_END = 10                # frames 1..FADE_END fade the "?" out
frames = []
for i in range(N_FRAMES):
    f = base.copy()
    if i > 0:
        a = min(1.0, i / FADE_END)              # 1 => fully white
        f[qys, qxs] = np.clip(qcol * (1 - a) + 255 * a, 0, 255).astype(np.uint8)
    if i >= FADE_END:
        p = (i - FADE_END) / (N_FRAMES - 1 - FADE_END)
        sel = order_t <= p + 1e-9
        f[tys[sel], txs[sel]] = colour[sel]
    frames.append(f)

os.makedirs(OUT_DIR, exist_ok=True)
tmp = os.path.join(OUT_DIR, "_frames")
os.makedirs(tmp, exist_ok=True)
for i, f in enumerate(frames):
    Image.fromarray(f).save(os.path.join(tmp, f"{i:03d}.png"))

subprocess.run([
    "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
    "-i", os.path.join(tmp, "%03d.png"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
    "-x264-params", "keyint=1", "-r", str(FPS), OUT,
], check=True)
for fn in os.listdir(tmp):
    os.remove(os.path.join(tmp, fn))
os.rmdir(tmp)
print("wrote", OUT, "answer colour", answer, "ref cell", ref, "shift", (dy, dx))
