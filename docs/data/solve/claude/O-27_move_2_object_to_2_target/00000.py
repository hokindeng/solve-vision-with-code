#!/usr/bin/env python3
"""Move two objects onto their matching dashed outlines along straight-line paths."""
import os, subprocess
import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 35, 16

img = np.array(Image.open(SRC).convert("RGB"))
H, W, _ = img.shape
bg = np.array([220, 220, 220], dtype=np.uint8)
fg = np.any(img != bg, axis=2)
lab, n = ndimage.label(fg)
sizes = ndimage.sum(fg, lab, range(1, n + 1))

# Solid objects are the large components; dashes are small.
obj_ids = [i + 1 for i, s in enumerate(sizes) if s > 1000]
dash_ids = [i + 1 for i, s in enumerate(sizes) if s <= 1000]


def hue_key(rgb):
    r, g, b = [int(v) for v in rgb]
    return (r > 40, g > 40, b > 40)


# Group dashes by color signature -> target outline masks
outlines = {}
for i in dash_ids:
    m = lab == i
    col = img[m][0]
    outlines.setdefault(hue_key(col), np.zeros_like(fg))
    outlines[hue_key(col)] |= m

moves = []  # (object mask, (dx, dy))
for oid in obj_ids:
    m = lab == oid
    ys, xs = np.where(m)
    # dominant colour of object
    cols, cnt = np.unique(img[m], axis=0, return_counts=True)
    key = hue_key(cols[np.argmax(cnt)])
    tgt = outlines[key]
    tys, txs = np.where(tgt)
    # Align the centroid of the object's boundary with the centroid of the
    # dashed outline. Both are boundary distributions of the same shape, so
    # this is robust even if the outline is drawn slightly larger.
    edge = m & ~ndimage.binary_erosion(m)
    ey, ex = np.where(edge)
    best = (int(round(txs.mean() - ex.mean())), int(round(tys.mean() - ey.mean())))
    moves.append((m, best))
    print(f"object {oid}: size {int(m.sum())}, move by {best}")

# Static background: everything except the moving objects
static = img.copy()
for m, _ in moves:
    static[m] = bg

os.makedirs(OUT_DIR, exist_ok=True)
ff = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
     "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-r", str(FPS), OUT],
    stdin=subprocess.PIPE)

for f in range(N_FRAMES):
    t = f / (N_FRAMES - 1)
    frame = static.copy()
    for m, (dx, dy) in moves:
        sx, sy = int(round(dx * t)), int(round(dy * t))
        ys, xs = np.where(m)
        ny, nx = ys + sy, xs + sx
        ok = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
        frame[ny[ok], nx[ok]] = img[ys[ok], xs[ok]]
    ff.stdin.write(frame.tobytes())
ff.stdin.close()
ff.wait()
print("wrote", OUT)
