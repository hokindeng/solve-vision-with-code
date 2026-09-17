#!/usr/bin/env python3
"""Regenerate /app/output/video.mp4: yellow books are inserted step by step into
the shelf gaps whose neighbouring book heights best match each book's height."""
import itertools
import math
import os
import subprocess

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

N_FRAMES = 22
FPS = 16
WHITE = np.array([255, 255, 255], np.uint8)
BLACK = np.array([0, 0, 0], np.uint8)
BORDER = 2


def detect_books(img):
    """Return list of dicts: x0, x1 (inclusive), top, bottom, fill colour."""
    a = img.astype(int)
    h, w, _ = a.shape
    # shelf = brown rows
    brown = (a[:, :, 0] > 60) & (a[:, :, 0] < 120) & (a[:, :, 1] < 70) & (a[:, :, 2] < 50)
    shelf_top = int(np.where(brown.any(1))[0].min())
    shelf_col = a[shelf_top + 1, 0].astype(np.uint8)
    probe = shelf_top - 8
    nonwhite = a[probe].sum(1) < 750
    xs = np.where(nonwhite)[0]
    books = []
    s = p = xs[0]
    for x in list(xs[1:]) + [None]:
        if x is None or x != p + 1:
            col = a[:, (s + p) // 2]
            top = int(np.where(col.sum(1) < 750)[0].min())
            fill = a[(top + shelf_top) // 2, (s + p) // 2].astype(np.uint8)
            books.append(dict(x0=int(s), x1=int(p), top=top, bottom=shelf_top, fill=fill))
            if x is not None:
                s = x
        if x is not None:
            p = x
    return books, shelf_top, shelf_col


def is_yellow(c):
    return c[0] > 140 and c[1] > 140 and c[2] < 80


def draw_book(frame, x0, top, w, hgt, fill):
    x1, y1 = x0 + w, top + hgt
    frame[top:y1, x0:x1] = BLACK
    frame[top + BORDER:y1 - BORDER, x0 + BORDER:x1 - BORDER] = fill


def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * t)


def main():
    img = np.array(Image.open(FIRST).convert("RGB"))
    books, shelf_top, shelf_col = detect_books(img)
    shelf_books = [b for b in books if not is_yellow(b["fill"])]
    yellow = [b for b in books if is_yellow(b["fill"])]
    bw = shelf_books[0]["x1"] - shelf_books[0]["x0"] + 1
    pitch = shelf_books[1]["x0"] - shelf_books[0]["x0"]
    hgt = lambda b: b["bottom"] - b["top"] + 1

    # gaps: spaces between consecutive shelf books wider than the normal pitch
    gaps = []
    for l, r in zip(shelf_books, shelf_books[1:]):
        if r["x0"] - l["x0"] > pitch + 2:
            gaps.append(dict(x=(l["x1"] + 1 + r["x0"] - bw) // 2, h=(hgt(l) + hgt(r)) / 2))

    # optimal assignment: minimise total |height - neighbourhood height|
    best = None
    for perm in itertools.permutations(range(len(gaps)), len(yellow)):
        cost = sum(abs(hgt(yb) - gaps[g]["h"]) for yb, g in zip(yellow, perm))
        if best is None or cost < best[0]:
            best = (cost, perm)
    assign = best[1]

    # background: first frame with yellow books erased
    bg = img.copy()
    for yb in yellow:
        bg[yb["top"]:shelf_top, yb["x0"]:yb["x1"] + 1] = WHITE
        bg[shelf_top, yb["x0"]:yb["x1"] + 1] = shelf_col

    lift_y = min(b["top"] for b in books) - 40  # travel height above tallest book
    # insertion order: left-most target gap first (gaps filled left to right)
    order = sorted(range(len(yellow)), key=lambda i: gaps[assign[i]]["x"])
    per = (N_FRAMES - 1) // len(yellow)
    frames = []
    for f in range(N_FRAMES):
        fr = bg.copy()
        for k, i in enumerate(order):
            yb = yellow[i]
            g = gaps[assign[i]]
            start = 1 + k * per
            end = start + per - 1
            if k == len(order) - 1:
                end = N_FRAMES - 1
            if f < start:
                t = 0.0
            elif f >= end:
                t = 1.0
            else:
                t = (f - start + 1) / (end - start + 1)
            sx, tx = yb["x0"], g["x"]
            # three-phase motion: lift (0-0.3), slide (0.3-0.7), lower (0.7-1)
            if t <= 0.3:
                u = ease(t / 0.3)
                x, y = sx, yb["top"] + (lift_y - yb["top"]) * u
            elif t <= 0.7:
                u = ease((t - 0.3) / 0.4)
                x, y = sx + (tx - sx) * u, lift_y
            else:
                u = ease((t - 0.7) / 0.3)
                x, y = tx, lift_y + (yb["top"] - lift_y) * u
            draw_book(fr, int(round(x)), int(round(y)), bw, hgt(yb), yb["fill"])
        frames.append(fr)

    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = os.path.join(OUT_DIR, "_frames")
    os.makedirs(tmp, exist_ok=True)
    for i, fr in enumerate(frames):
        Image.fromarray(fr).save(os.path.join(tmp, f"{i:03d}.png"))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp, "%03d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "12", "-preset", "slow", OUT], check=True)
    for fn in os.listdir(tmp):
        os.remove(os.path.join(tmp, fn))
    os.rmdir(tmp)
    print("assignment (yellow idx -> gap x, target height):",
          [(i, gaps[assign[i]]["x"], hgt(yellow[i])) for i in range(len(yellow))])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
