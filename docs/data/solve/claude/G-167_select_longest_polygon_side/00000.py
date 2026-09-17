#!/usr/bin/env python3
"""Generate the video: compare all polygon edges step by step, then mark the
longest edge with a small red circle at its midpoint."""
import os
import subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS = 16
N_FRAMES = 25
N_SIDES = 8
SS = 4  # supersampling for antialiased overlays

base = np.array(Image.open(FIRST).convert("RGB"))
H, W = base.shape[:2]


def find_vertices(img, n_sides):
    mask = (img.min(axis=2) < 240).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    lo, hi = 0.0, 0.2
    best = None
    for _ in range(60):  # binary search epsilon to get exactly n_sides vertices
        eps = (lo + hi) / 2
        approx = cv2.approxPolyDP(cnt, eps * peri, True)
        if len(approx) > n_sides:
            lo = eps
        elif len(approx) < n_sides:
            hi = eps
        else:
            best = approx
            break
    if best is None:
        raise RuntimeError("could not reduce contour to %d vertices" % n_sides)
    return best.reshape(-1, 2).astype(float)


verts = find_vertices(base, N_SIDES)
edges = [(verts[i], verts[(i + 1) % N_SIDES]) for i in range(N_SIDES)]
lengths = [float(np.hypot(*(b - a))) for a, b in edges]
longest = int(np.argmax(lengths))
mid = (edges[longest][0] + edges[longest][1]) / 2.0
print("vertices:\n", verts)
print("edge lengths:", [round(l, 1) for l in lengths])
print("longest edge index:", longest, "midpoint:", mid)


def overlay(base_img, draw_fn):
    """Draw antialiased overlay via supersampled RGBA layer composited on base."""
    layer = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    draw_fn(d)
    layer = layer.resize((W, H), Image.LANCZOS)
    out = Image.fromarray(base_img.copy()).convert("RGBA")
    out.alpha_composite(layer)
    return np.array(out.convert("RGB"))


def line(d, a, b, color, width):
    d.line([tuple(a * SS), tuple(b * SS)], fill=color, width=int(width * SS))


def dot(d, c, r, color):
    x, y = c * SS
    d.ellipse([x - r * SS, y - r * SS, x + r * SS, y + r * SS], fill=color)


BLUE = (37, 99, 235, 255)
ORANGE = (245, 158, 11, 255)
RED = (220, 38, 38, 255)
CIRCLE_R = 9

frames = [base.copy()]  # frame 0 == first_frame exactly

# Step phase: visit each edge in turn (2 frames each), keep best-so-far marked.
best_so_far = None
for i in range(N_SIDES):
    if best_so_far is None or lengths[i] > lengths[best_so_far]:
        best_so_far = i
    for _ in range(2):
        def fn(d, i=i, bsf=best_so_far):
            if bsf != i:
                line(d, *edges[bsf], ORANGE, 5)
            line(d, *edges[i], BLUE, 5)
        frames.append(overlay(base, fn))

# Result phase: only the longest edge highlighted, then the red circle grows in
# while the highlight fades away; final frame = base + red circle only.
for k in range(3):
    frames.append(overlay(base, lambda d: line(d, *edges[longest], ORANGE, 5)))

remaining = N_FRAMES - len(frames)
for k in range(remaining):
    t = (k + 1) / remaining
    alpha = int(round(255 * (1 - t)))
    r = CIRCLE_R * min(1.0, t * 1.5)

    def fn(d, alpha=alpha, r=r):
        if alpha > 0:
            line(d, *edges[longest], ORANGE[:3] + (alpha,), 5)
        dot(d, mid, r, RED)
    frames.append(overlay(base, fn))

assert len(frames) == N_FRAMES

os.makedirs(OUT_DIR, exist_ok=True)
tmp = os.path.join(OUT_DIR, "_frames")
os.makedirs(tmp, exist_ok=True)
for i, f in enumerate(frames):
    Image.fromarray(f).save(os.path.join(tmp, "f%03d.png" % i))
subprocess.run([
    "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
    "-i", os.path.join(tmp, "f%03d.png"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
    "-r", str(FPS), OUT,
], check=True)
for name in os.listdir(tmp):
    os.remove(os.path.join(tmp, name))
os.rmdir(tmp)
print("wrote", OUT)
