#!/usr/bin/env python3
"""Generate the bookshelf insertion video from first_frame.png."""
import os, subprocess, tempfile
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT = os.path.join(HERE, "output", "video.mp4")
N_FRAMES, FPS = 24, 16

base = np.array(Image.open(FIRST).convert("RGB"))
H, W, _ = base.shape
WHITE = np.array([255, 255, 255], np.uint8)
YELLOW = np.array([150, 150, 28])

# ---- detect books (columns of non-white pixels at a body row) ----
ROW = 480
cols = np.where((base[ROW] != 255).any(axis=1))[0]
runs, s, p = [], cols[0], cols[0]
for c in cols[1:]:
    if c != p + 1:
        runs.append((s, p)); s = c
    p = c
runs.append((s, p))

books = []
for x0, x1 in runs:
    xm = (x0 + x1) // 2
    ys = np.where((base[:, xm] != 255).any(axis=1))[0]
    top = int(ys.min())
    color = base[(top + ROW) // 2, xm]
    books.append(dict(x0=int(x0), x1=int(x1), top=top,
                      yellow=bool(np.abs(color.astype(int) - YELLOW).sum() < 30)))

# bottom of books = last black row of border going down from the body row
def book_bottom(x):
    y = ROW
    while (base[y, x] != 255).any() and y < H - 1:
        y += 1
    # walk down through the black border (border overlaps the shelf top)
    y = ROW
    while tuple(base[y, x]) != (52, 41, 22) and y < H - 1:
        y += 1
    return y - 1
BOTTOM = book_bottom(books[0]["x0"])           # last row of the book (border)
SHELF_COLOR = base[BOTTOM + 1, books[0]["x0"]].copy()
SHELF_TOP = BOTTOM  # the book's bottom border row overlaps the shelf's first row

shelf_books = [b for b in books if not b["yellow"]]
yellow_books = [b for b in books if b["yellow"]]

# ---- gaps between shelf books (wide enough for a book) ----
book_w = shelf_books[0]["x1"] - shelf_books[0]["x0"] + 1
gaps = []
for a, b in zip(shelf_books, shelf_books[1:]):
    free = b["x0"] - a["x1"] - 1
    if free >= book_w + 2:
        gaps.append(dict(left=a, right=b, x0=a["x1"] + 1, x1=b["x0"] - 1,
                         height=(a["top"] + b["top"]) / 2))

# ---- assign each yellow book to the gap with closest surrounding height ----
assign = {}
remaining = list(range(len(gaps)))
for i, yb in sorted(enumerate(yellow_books), key=lambda t: t[1]["x0"]):
    best = min(remaining, key=lambda g: abs(gaps[g]["height"] - yb["top"]))
    assign[i] = best
    remaining.remove(best)

# ---- patches & background with the yellow books removed ----
bg = base.copy()
patches = []
for yb in yellow_books:
    patch = base[yb["top"]:BOTTOM + 1, yb["x0"]:yb["x1"] + 1].copy()
    patches.append(patch)
    bg[yb["top"]:SHELF_TOP, yb["x0"]:yb["x1"] + 1] = WHITE
    bg[SHELF_TOP:BOTTOM + 1, yb["x0"]:yb["x1"] + 1] = SHELF_COLOR

def paste(img, patch, x, y):
    h, w, _ = patch.shape
    img[y:y + h, x:x + w] = patch

def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)

# ---- motion plan: each book lifts, slides left, drops (sequential) ----
lift_top = min(b["top"] for b in books) - 40
order = sorted(range(len(yellow_books)), key=lambda i: yellow_books[i]["x0"])
n_anim = N_FRAMES - 1
seg = n_anim // len(order)
plans = []
for k, i in enumerate(order):
    yb = yellow_books[i]; g = gaps[assign[i]]
    gx = (g["x0"] + g["x1"] + 1) // 2 - book_w // 2
    f0 = 1 + k * seg
    f1 = f0 + seg if k < len(order) - 1 else N_FRAMES
    plans.append(dict(i=i, sx=yb["x0"], sy=yb["top"], tx=gx, ty=yb["top"], f0=f0, f1=f1))

def position(plan, f):
    if f < plan["f0"]:
        return plan["sx"], plan["sy"]
    if f >= plan["f1"] - 1:
        return plan["tx"], plan["ty"]
    t = (f - plan["f0"]) / (plan["f1"] - 1 - plan["f0"])
    # three phases: lift 25%, slide 50%, drop 25%
    if t < 0.25:
        u = ease(t / 0.25)
        return plan["sx"], int(round(plan["sy"] + (lift_top - plan["sy"]) * u))
    if t < 0.75:
        u = ease((t - 0.25) / 0.5)
        return int(round(plan["sx"] + (plan["tx"] - plan["sx"]) * u)), lift_top
    u = ease((t - 0.75) / 0.25)
    return plan["tx"], int(round(lift_top + (plan["ty"] - lift_top) * u))

frames = []
for f in range(N_FRAMES):
    if f == 0:
        frames.append(base.copy()); continue
    img = bg.copy()
    for plan in plans:
        x, y = position(plan, f)
        paste(img, patches[plan["i"]], x, y)
    frames.append(img)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with tempfile.TemporaryDirectory() as td:
    for k, fr in enumerate(frames):
        Image.fromarray(fr).save(os.path.join(td, f"{k:04d}.png"))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", os.path.join(td, "%04d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT], check=True)
print("wrote", OUT, "frames:", len(frames), "assignment:", {yellow_books[i]["x0"]: gaps[g]["x0"] for i, g in assign.items()})
