#!/usr/bin/env python3
"""Animate pink books sliding into the bookshelf gaps whose neighbours best match their height."""
import os
import subprocess
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 24

PINK = np.array([135, 25, 80], np.uint8)
TAN = np.array([187, 173, 72], np.uint8)
SHELF = np.array([102, 71, 49], np.uint8)
WHITE = np.array([255, 255, 255], np.uint8)
BLACK = np.array([0, 0, 0], np.uint8)
SHELF_Y = 512          # first shelf row; book bottom outline overlaps it
FILL_BOTTOM = 510      # last fill row of every book
BORDER = 2
BOOK_W = 23            # fill width
PITCH = 32


def find_books(img, color):
    """Return list of (x0_fill, top_fill) for each column run matching color."""
    mask = (img == color).all(2)
    cols = np.where(mask.any(0))[0]
    runs, s, p = [], cols[0], cols[0]
    for x in cols[1:]:
        if x != p + 1:
            runs.append((s, p)); s = x
        p = x
    runs.append((s, p))
    out = []
    for x0, x1 in runs:
        rows = np.where(mask[:, x0:x1 + 1].any(1))[0]
        out.append((int(x0), int(rows.min())))
    return out


def draw_book(img, x0, top, color):
    """Draw a book with fill at columns x0..x0+22, rows top..510, 2px black outline."""
    x0, top = int(round(x0)), int(round(top))
    y_lo, y_hi = top - BORDER, SHELF_Y + 1          # outline rows [top-2, 512]
    x_lo, x_hi = x0 - BORDER, x0 + BOOK_W + BORDER  # outline cols [x0-2, x0+25]
    img[y_lo:y_hi, x_lo:x_hi] = BLACK
    img[top:FILL_BOTTOM + 1, x0:x0 + BOOK_W] = color


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1))


def main():
    base = np.array(Image.open(SRC).convert("RGB"))
    tan = find_books(base, TAN)
    pink = find_books(base, PINK)

    # Erase pink books from the background (white above shelf, shelf row restored).
    bg = base.copy()
    for x0, top in pink:
        bg[:SHELF_Y, x0 - BORDER:x0 + BOOK_W + BORDER] = WHITE
        bg[SHELF_Y, x0 - BORDER:x0 + BOOK_W + BORDER] = SHELF

    # Gaps: consecutive tan books more than one pitch apart -> one empty slot each.
    gaps = []
    for (xa, ta), (xb, tb) in zip(tan, tan[1:]):
        n = round((xb - xa) / PITCH) - 1
        for k in range(1, n + 1):
            gaps.append((xa + k * PITCH, (ta + tb) / 2.0))

    # Assign each pink book to the gap with closest neighbour height (greedy, unique).
    free = list(range(len(gaps)))
    targets = []
    for x0, top in pink:
        j = min(free, key=lambda g: abs(gaps[g][1] - top))
        free.remove(j)
        targets.append(gaps[j][0])

    lift_top = min(t for _, t in tan + pink) - 40  # travel height above all books

    frames = [base.copy()]
    n_move = N_FRAMES - 1
    seg = n_move / len(pink)
    for f in range(1, N_FRAMES):
        img = bg.copy()
        moving = None
        for i, ((sx, top), tx) in enumerate(zip(pink, targets)):
            t = (f - i * seg) / seg  # progress of book i in [0, 1]
            if t <= 0:
                draw_book(img, sx, top, PINK)
            elif t >= 1:
                draw_book(img, tx, top, PINK)
            else:
                # lift (0-0.25), slide (0.25-0.75), drop (0.75-1)
                if t < 0.25:
                    x, y = sx, top + (lift_top - top) * ease(t / 0.25)
                elif t < 0.75:
                    x, y = sx + (tx - sx) * ease((t - 0.25) / 0.5), lift_top
                else:
                    x, y = tx, lift_top + (top - lift_top) * ease((t - 0.75) / 0.25)
                moving = (x, y)
        if moving:
            draw_book(img, moving[0], moving[1], PINK)
        frames.append(img)

    assert (frames[0] == base).all()
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = os.path.join(OUT_DIR, "frames")
    os.makedirs(tmp, exist_ok=True)
    for i, fr in enumerate(frames):
        Image.fromarray(fr).save(os.path.join(tmp, f"{i:03d}.png"))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", os.path.join(tmp, "%03d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "12", OUT], check=True)
    for fn in os.listdir(tmp):
        os.remove(os.path.join(tmp, fn))
    os.rmdir(tmp)
    print("assignments:", [(sx, tx) for (sx, _), tx in zip(pink, targets)])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
