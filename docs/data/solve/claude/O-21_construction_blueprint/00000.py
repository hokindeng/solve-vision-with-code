#!/usr/bin/env python3
"""Generate the puzzle-check video from first_frame.png.

Scene (measured from first_frame.png):
  * Structure grid: cell pitch 52 px, origin (486, 331); each cell is a 51x51
    rect (fill (70,130,180), 1 px outline (50,100,140)) followed by 1 px gap.
  * Gap (red dashed cells): grid cells (-1,0),(-1,1),(-2,2),(-1,2),(0,2)
    (columns relative to the leftmost blue column x=486).
  * Candidate boxes: 186x186 squares, outer border 1 px gray(180) at
    x = 50 + 246*i .. 235 + 246*i, y = 829 .. 1014, fill gray(230).
  * Candidate pieces: cell pitch 28 px (27 px cells), top-left of piece
    bounding box at y = 879 and x as measured per candidate.
"""
import os
import shutil
import subprocess
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

FPS = 16
N_FRAMES = 99
W = H = 1024

BG = (245, 245, 245)
BLUE = (70, 130, 180)
BLUE_EDGE = (50, 100, 140)
RED_DASH = (255, 100, 100)

# structure grid
CELL = 52           # pitch
CELL_DRAW = 50      # rect from x to x+50 inclusive -> 51 px
GX0, GY0 = 486, 331
GAP_CELLS = [(-1, 0), (-1, 1), (-2, 2), (-1, 2), (0, 2)]
GAP_ORIGIN = (-2, 0)   # top-left of gap bounding box in grid coords (3 cols x 3 rows)

# candidate boxes
BOX_X0 = [50 + 246 * i for i in range(4)]
BOX_Y0, BOX_SIZE = 829, 185   # inclusive extents: x0..x0+185
CCELL = 28
CCELL_DRAW = 26
# piece cells (col,row) relative to piece bbox, and bbox top-left pixel
CANDS = [
    dict(cells=[(0, 0), (1, 0), (1, 1), (2, 1), (3, 1), (2, 2)], px=(86, 879)),
    dict(cells=[(2, 0), (0, 1), (1, 1), (2, 1), (2, 2)], px=(346, 879)),
    dict(cells=[(0, 0), (0, 1), (0, 2), (1, 2)], px=(606, 879)),
    dict(cells=[(1, 0), (1, 1), (0, 2), (1, 2), (2, 2)], px=(838, 879)),
]
GAP_NORM = sorted((c - GAP_ORIGIN[0], r - GAP_ORIGIN[1]) for c, r in GAP_CELLS)
MATCH = [sorted(c["cells"]) == GAP_NORM for c in CANDS]
assert MATCH.count(True) == 1
ANSWER = MATCH.index(True)

# colours for judging / highlighting
HILITE = (255, 165, 0)
GREEN_FILL, GREEN = (205, 236, 210), (40, 160, 70)
RED_FILL, RED = (246, 210, 210), (215, 55, 55)

# ---- timeline (frame indices) ----------------------------------------------
PER = 18                       # frames per candidate
T_HL, T_PREV = 4, 8            # highlight-only, then preview, rest = judged
CAND_START = 1                 # frame 0 is the untouched first frame
MOVE_START = CAND_START + 4 * PER      # 73
MOVE_LEN = 18                          # 73..90
FINAL_START = MOVE_START + MOVE_LEN    # 91..98 hold


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1))


def draw_cell(d, x, y, size, fill=BLUE, edge=BLUE_EDGE):
    d.rectangle([x, y, x + size, y + size], fill=fill, outline=edge, width=1)


def cell_px(c, r):
    return GX0 + c * CELL, GY0 + r * CELL


def erase_dashes(img):
    a = np.array(img)
    m = (a == np.array(RED_DASH, dtype=a.dtype)).all(2)
    a[m] = BG
    return Image.fromarray(a)


def draw_gap_filled(d):
    for c, r in GAP_CELLS:
        x, y = cell_px(c, r)
        draw_cell(d, x, y, CELL_DRAW)


def box_rect(i):
    x0 = BOX_X0[i]
    return [x0, BOX_Y0, x0 + BOX_SIZE, BOX_Y0 + BOX_SIZE]


def draw_piece_in_box(d, i):
    px, py = CANDS[i]["px"]
    for c, r in CANDS[i]["cells"]:
        draw_cell(d, px + c * CCELL, py + r * CCELL, CCELL_DRAW)


def restyle_box(img, i, fill, border, keep_piece=True):
    """Recolour a candidate box (fill + border), keep label & piece intact."""
    a = np.array(img)
    x0, y0, x1, y1 = box_rect(i)
    sub = a[y0:y1 + 1, x0:x1 + 1]
    inner = (sub == 230).all(2)
    sub[inner] = fill
    a[y0:y1 + 1, x0:x1 + 1] = sub
    img = Image.fromarray(a)
    d = ImageDraw.Draw(img)
    d.rectangle([x0, y0, x1, y1], outline=border, width=2)
    if not keep_piece:
        px, py = CANDS[i]["px"]
        for c, r in CANDS[i]["cells"]:
            d.rectangle([px + c * CCELL, py + r * CCELL,
                         px + c * CCELL + CCELL_DRAW, py + r * CCELL + CCELL_DRAW], fill=fill)
    return img


def draw_mark(d, i, ok):
    x0, y0, x1, y1 = box_rect(i)
    cx, cy = x1 - 26, y0 + 26
    w = 5
    if ok:
        d.line([(cx - 13, cy), (cx - 4, cy + 10), (cx + 14, cy - 12)], fill=GREEN, width=w, joint="curve")
    else:
        d.line([(cx - 11, cy - 11), (cx + 11, cy + 11)], fill=RED, width=w)
        d.line([(cx - 11, cy + 11), (cx + 11, cy - 11)], fill=RED, width=w)


def draw_highlight(d, i):
    x0, y0, x1, y1 = box_rect(i)
    d.rectangle([x0, y0, x1, y1], outline=HILITE, width=4)


def preview_piece(img, i, alpha):
    """Ghost of candidate i placed at the gap bounding box."""
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    gap_set = set(GAP_CELLS)
    for c, r in CANDS[i]["cells"]:
        gc, gr = c + GAP_ORIGIN[0], r + GAP_ORIGIN[1]
        x, y = cell_px(gc, gr)
        fits = (gc, gr) in gap_set
        col = BLUE if fits else RED
        edge = BLUE_EDGE if fits else (150, 30, 30)
        a = int(255 * alpha)
        d.rectangle([x, y, x + CELL_DRAW, y + CELL_DRAW], fill=col + (a,), outline=edge + (a,), width=2)
    base = img.convert("RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")


def moving_piece(img, i, t):
    """Piece i interpolated from its box (small) to the gap (large)."""
    px, py = CANDS[i]["px"]
    gx, gy = cell_px(*GAP_ORIGIN)
    # travel left first, then rise, so the piece never crosses the structure
    sx = ease(min(1.0, t / 0.6))
    sy = ease(max(0.0, (t - 0.25) / 0.75))
    s = ease(t)
    ox = px + (gx - px) * sx
    oy = py + (gy - py) * sy
    pitch = CCELL + (CELL - CCELL) * s
    size = CCELL_DRAW + (CELL_DRAW - CCELL_DRAW) * s
    d = ImageDraw.Draw(img)
    for c, r in CANDS[i]["cells"]:
        x = ox + c * pitch
        y = oy + r * pitch
        d.rectangle([round(x), round(y), round(x + size), round(y + size)], fill=BLUE, outline=BLUE_EDGE, width=1)
    return img


def render_frame(k, base):
    img = base.copy()
    if k == 0:
        return img

    # persistent judgements
    judged = []
    for i in range(4):
        end = CAND_START + i * PER + T_HL + T_PREV
        if k >= end:
            judged.append(i)

    # which candidate is active
    active = None
    if CAND_START <= k < MOVE_START:
        active = (k - CAND_START) // PER
        phase = (k - CAND_START) % PER

    if k >= MOVE_START:
        img = erase_dashes(img)

    for i in judged:
        ok = MATCH[i]
        keep = not (ok and k >= MOVE_START)
        img = restyle_box(img, i, GREEN_FILL if ok else RED_FILL, GREEN if ok else RED, keep_piece=keep)
        draw_mark(ImageDraw.Draw(img), i, ok)

    if active is not None:
        if phase < T_HL + T_PREV:
            draw_highlight(ImageDraw.Draw(img), active)
        if T_HL <= phase < T_HL + T_PREV:
            # fade in the ghost preview
            alpha = 0.25 + 0.4 * ease((phase - T_HL + 1) / 3.0)
            img = preview_piece(img, active, alpha)
        elif phase >= T_HL + T_PREV and active == ANSWER:
            # keep showing the fitting preview for the matching piece
            img = preview_piece(img, active, 0.65)

    if MOVE_START <= k < FINAL_START:
        t = (k - MOVE_START + 1) / MOVE_LEN
        img = moving_piece(img, ANSWER, t)
    elif k >= FINAL_START:
        draw_gap_filled(ImageDraw.Draw(img))
    return img


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = Image.open(FIRST).convert("RGB")
    frames_dir = os.path.join(OUT_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for k in range(N_FRAMES):
        render_frame(k, base).save(os.path.join(frames_dir, f"{k:04d}.png"))
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(frames_dir, "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT,
    ]
    subprocess.run(cmd, check=True)
    shutil.rmtree(frames_dir, ignore_errors=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
