#!/usr/bin/env python3
"""Animate placing the waiting blue books into the shelf gaps whose neighbour
heights best match each book, then encode to /app/output/video.mp4."""
import math
import os
import subprocess

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 24

WHITE = np.array([255, 255, 255])
BLUE = np.array([32, 101, 171])
SHELF = np.array([92, 60, 40])


def find_books(img):
    """Return list of (x0, x1, top, bottom, fill) for every book on the shelf row."""
    shelf_rows = np.where((img == SHELF).all(2).sum(1) > 400)[0]
    shelf_top = shelf_rows.min()
    y = shelf_top - 3
    nonwhite = ~(img[y] == WHITE).all(1)
    books, x0 = [], None
    for x in range(img.shape[1] + 1):
        on = x < img.shape[1] and nonwhite[x]
        if on and x0 is None:
            x0 = x
        elif not on and x0 is not None:
            xm = (x0 + x - 1) // 2
            col = img[:shelf_top, xm]
            top = int(np.where(~(col == WHITE).all(1))[0].min())
            fill = img[(top + shelf_top) // 2, xm]
            books.append(dict(x0=x0, x1=x - 1, top=top, bottom=shelf_top, fill=tuple(fill)))
            x0 = None
    return books, shelf_top


def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * t)


def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    books, shelf_top = find_books(base)
    height = lambda b: b["bottom"] - b["top"]
    brown = [b for b in books if b["fill"] != tuple(BLUE)]
    blue = [b for b in books if b["fill"] == tuple(BLUE)]
    book_w = brown[0]["x1"] - brown[0]["x0"] + 1

    # Gaps between consecutive brown books wide enough for a book.
    gaps = []
    for a, b in zip(brown, brown[1:]):
        space = b["x0"] - a["x1"] - 1
        if space >= book_w + 4:
            gaps.append(dict(x0=(a["x1"] + b["x0"] + 1) // 2 - book_w // 2,
                             target_h=(height(a) + height(b)) / 2))

    # Assign each blue book to the gap with the closest neighbour height.
    assignments = []
    free = list(range(len(gaps)))
    for b in sorted(blue, key=lambda b: b["x0"]):
        gi = min(free, key=lambda i: abs(gaps[i]["target_h"] - height(b)))
        free.remove(gi)
        assignments.append((b, gaps[gi]))

    # Cut out the blue book sprites and clear their original spots.
    canvas = base.copy()
    sprites = []
    for b, g in assignments:
        patch = base[b["top"]:b["bottom"] + 1, b["x0"]:b["x1"] + 1].copy()
        canvas[b["top"]:b["bottom"], b["x0"]:b["x1"] + 1] = WHITE
        canvas[b["bottom"], b["x0"]:b["x1"] + 1] = SHELF  # shelf edge under the border
        sprites.append((patch, b["x0"], b["top"], g["x0"]))

    # Schedule: frame 0 is untouched; each book gets an equal share of the rest.
    n_move = N_FRAMES - 1
    per = n_move / len(sprites)
    lift = 130
    frames = []
    for f in range(N_FRAMES):
        frame = canvas.copy()
        for i, (patch, sx, sy, tx) in enumerate(sprites):
            start, end = 1 + i * per, 1 + (i + 1) * per
            t = 0.0 if f < start else 1.0 if f >= end - 1 else (f - start) / (end - 1 - start)
            t = min(max(t, 0.0), 1.0)
            x = round(sx + (tx - sx) * ease(t))
            y = round(sy - lift * math.sin(math.pi * t))
            h, w = patch.shape[:2]
            frame[y:y + h, x:x + w] = patch
        frames.append(frame)

    os.makedirs(OUT_DIR, exist_ok=True)
    raw = np.concatenate(frames).tobytes()
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{base.shape[1]}x{base.shape[0]}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-qp", "0", "-pix_fmt", "yuv420p",
           "-r", str(FPS), OUT]
    subprocess.run(cmd, input=raw, check=True)
    Image.fromarray(frames[-1]).save(os.path.join(OUT_DIR, "last_frame.png"))
    print(f"wrote {OUT} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
