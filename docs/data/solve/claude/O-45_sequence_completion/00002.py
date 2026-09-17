#!/usr/bin/env python3
"""Complete the shape_cycle sequence: circle, circle, heptagon, circle, circle, ? -> heptagon.

The '?' fades out, then a heptagon (identical to the one at position 3) grows into
the last slot. All other pixels are left exactly as in first_frame.png.
"""
import subprocess, os
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 25, 16

base = np.array(Image.open(FIRST).convert("RGB"))
H, W, _ = base.shape
nonwhite = np.any(base != 255, axis=2)

# --- segment elements by column projection -------------------------------
cols = nonwhite.any(axis=0)
segs, start = [], None
for x in range(W):
    if cols[x] and start is None:
        start = x
    if not cols[x] and start is not None:
        segs.append((start, x - 1)); start = None
if start is not None:
    segs.append((start, W - 1))

def bbox(seg):
    s, e = seg
    ys = np.where(nonwhite[:, s:e + 1].any(axis=1))[0]
    return s, e, ys.min(), ys.max()

boxes = [bbox(s) for s in segs]
qbox = boxes[-1]                       # the question mark
shapes = boxes[:-1]

# classify shapes: circle => thick (4px) black stroke on the center row; polygon => 1px
def kind(b):
    x0, x1, y0, y1 = b
    cy = (y0 + y1) // 2
    row = base[cy, x0:x1 + 1]
    n_black = 0
    for p in row:
        if (p == 0).all(): n_black += 1
        else: break
    return "circle" if n_black >= 3 else "polygon"

kinds = [kind(b) for b in shapes]
# find the cycle period
period = next(p for p in range(1, len(kinds) + 1)
              if all(kinds[i] == kinds[i % p] for i in range(len(kinds))))
answer_kind = kinds[len(kinds) % period]
src_idx = kinds.index(answer_kind)
sx0, sx1, sy0, sy1 = shapes[src_idx]
patch = base[sy0:sy1 + 1, sx0:sx1 + 1].copy()
ph, pw = patch.shape[:2]
patch_mask = (np.any(patch != 255, axis=2) * 255).astype(np.uint8)  # shape coverage

# target center: extrapolate the horizontal spacing
centers = [((b[0] + b[1]) / 2) for b in shapes]
step = (centers[-1] - centers[0]) / (len(centers) - 1)
tcx = int(round(centers[-1] + step))
tcy = int(round((sy0 + sy1) / 2))
tx0 = tcx - (sx0 + sx1) // 2 + sx0     # keep same pixel phase as the source patch
ty0 = sy0

# region of the '?' with background cleared
qx0, qx1, qy0, qy1 = qbox
cleared = base.copy()
cleared[qy0:qy1 + 1, qx0:qx1 + 1] = 255

def ease(t):
    return 1 - (1 - t) ** 3

def frame(i):
    t = i / (N_FRAMES - 1)
    img = base.astype(np.float32).copy()
    # phase 1: fade out the question mark (t in [0, 0.45])
    a = min(1.0, t / 0.45)
    reg = base[qy0:qy1 + 1, qx0:qx1 + 1].astype(np.float32)
    img[qy0:qy1 + 1, qx0:qx1 + 1] = reg * (1 - a) + 255 * a
    # phase 2: grow heptagon (t in [0.4, 1])
    if t >= 0.4:
        s = ease(min(1.0, (t - 0.4) / 0.6))
        if i == N_FRAMES - 1:
            img[ty0:ty0 + ph, tx0:tx0 + pw] = patch      # exact copy
        elif s > 0.02:
            sw, sh = max(1, int(round(pw * s))), max(1, int(round(ph * s)))
            small = np.array(Image.fromarray(patch).resize((sw, sh), Image.LANCZOS)).astype(np.float32)
            alpha = np.array(Image.fromarray(patch_mask).resize((sw, sh), Image.LANCZOS)).astype(np.float32) / 255.0
            alpha = np.clip(alpha, 0, 1)[..., None] * min(1.0, s * 3)
            ox = tx0 + (pw - sw) // 2
            oy = ty0 + (ph - sh) // 2
            dst = img[oy:oy + sh, ox:ox + sw]
            img[oy:oy + sh, ox:ox + sw] = dst * (1 - alpha) + small * alpha
    return np.clip(img + 0.5, 0, 255).astype(np.uint8)

os.makedirs(OUT_DIR, exist_ok=True)
frames_dir = os.path.join(OUT_DIR, "frames")
os.makedirs(frames_dir, exist_ok=True)
for i in range(N_FRAMES):
    Image.fromarray(frame(i)).save(os.path.join(frames_dir, f"{i:03d}.png"))

subprocess.run([
    "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
    "-i", os.path.join(frames_dir, "%03d.png"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
    "-r", str(FPS), OUT,
], check=True)
print("kinds:", kinds, "period:", period, "answer:", answer_kind, "target center:", (tcx, tcy))
print("wrote", OUT)
