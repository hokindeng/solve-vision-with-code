#!/usr/bin/env python3
"""Animate purple books being inserted into bookshelf gaps by height match."""
import os, subprocess, tempfile
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT = os.path.join(HERE, "output", "video.mp4")
N_FRAMES, FPS = 24, 16

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
SHELF = (60, 49, 36)
ORANGE = (197, 95, 25)
PURPLE = (75, 18, 132)
SHELF_TOP = 512          # book outlines end on this row (overlapping shelf top)
BOOK_W = 25
PITCH = 30


def detect_books(img):
    """Return list of dicts {x, top, h, color} for every book above the shelf."""
    a = np.array(img.convert("RGB")).astype(int)
    region = a[:SHELF_TOP]
    nw = (region != 255).any(2)
    xs = np.where(nw.any(0))[0]
    runs, s, p = [], xs[0], xs[0]
    for x in xs[1:]:
        if x != p + 1:
            runs.append((s, p)); s = x
        p = x
    runs.append((s, p))
    books = []
    for s, e in runs:
        ys = np.where(nw[:, s:e + 1].any(1))[0]
        top = int(ys.min())
        col = tuple(int(v) for v in a[(top + SHELF_TOP) // 2, (s + e) // 2])
        books.append(dict(x=int(s), top=top, h=SHELF_TOP - top, color=col))
    return books


def find_gaps(fixed):
    """Single-book-wide gaps between consecutive fixed books -> (x, neighbour heights)."""
    gaps = []
    for a, b in zip(fixed, fixed[1:]):
        space = b["x"] - (a["x"] + BOOK_W)
        if space >= BOOK_W + 2 * (PITCH - BOOK_W) - 1:
            gaps.append(dict(x=a["x"] + PITCH, hs=(a["h"], b["h"])))
    return gaps


def assign(purples, gaps):
    """Greedy best-fit: each purple book goes to the gap whose neighbour heights are closest."""
    free = list(range(len(gaps)))
    result = {}
    # process pairs by best cost first so the globally best matches happen
    pairs = sorted(((abs(g["hs"][0] - p["h"]) + abs(g["hs"][1] - p["h"]), pi, gi)
                    for pi, p in enumerate(purples) for gi, g in enumerate(gaps)))
    used_p = set()
    for cost, pi, gi in pairs:
        if pi in used_p or gi not in free:
            continue
        result[pi] = gi; used_p.add(pi); free.remove(gi)
    return result


def draw_book(draw, x, top, color):
    draw.rectangle([x, top, x + BOOK_W - 1, SHELF_TOP], fill=color, outline=BLACK, width=2)


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def main():
    first = Image.open(FIRST).convert("RGB")
    books = detect_books(first)
    fixed = [b for b in books if b["color"] != PURPLE]
    purples = [b for b in books if b["color"] == PURPLE]
    gaps = find_gaps(fixed)
    target = assign(purples, gaps)

    # background: first frame with purple books erased (white above shelf, shelf row restored)
    bg = first.copy()
    d = ImageDraw.Draw(bg)
    for p in purples:
        d.rectangle([p["x"], p["top"], p["x"] + BOOK_W - 1, SHELF_TOP - 1], fill=WHITE)
        d.line([p["x"], SHELF_TOP, p["x"] + BOOK_W - 1, SHELF_TOP], fill=SHELF)

    lift_top = min(b["top"] for b in books) - 20   # clearance above the tallest book
    n_move = N_FRAMES - 1
    per = n_move / len(purples)                    # frames per book, last ends on final frame

    frames = []
    for f in range(N_FRAMES):
        im = bg.copy()
        d = ImageDraw.Draw(im)
        for i, p in enumerate(purples):
            gx = gaps[target[i]]["x"]
            prog = (f - (1 + i * per)) / per if f >= 1 else -1
            if prog <= 0:
                x, top = p["x"], p["top"]
            elif prog >= 1:
                x, top = gx, p["top"]
            else:
                # phases: lift 0-0.25, slide 0.25-0.75, drop 0.75-1
                lift = p["top"] - (p["top"] - lift_top) * ease(prog / 0.25) if prog < 0.25 else lift_top
                if prog > 0.75:
                    lift = lift_top + (p["top"] - lift_top) * ease((prog - 0.75) / 0.25)
                sx = ease((prog - 0.25) / 0.5)
                x = p["x"] + (gx - p["x"]) * sx
                top = lift
            draw_book(d, int(round(x)), int(round(top)), PURPLE)
        frames.append(im)

    assert np.array_equal(np.array(frames[0]), np.array(first)), "frame 0 must equal first_frame"

    with tempfile.TemporaryDirectory() as td:
        for i, fr in enumerate(frames):
            fr.save(os.path.join(td, f"{i:04d}.png"))
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                        "-i", os.path.join(td, "%04d.png"), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "15", "-r", str(FPS), OUT], check=True)
    print("wrote", OUT, "assignment:", {purples[i]["h"]: gaps[g]["x"] for i, g in target.items()})


if __name__ == "__main__":
    main()
