#!/usr/bin/env python3
"""Generate video: heart shrinks (like the example hexagon), then becomes outline-only."""
import os, subprocess, shutil
import numpy as np, cv2
from PIL import Image

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 16, 16

GREEN = np.array([170, 191, 66], dtype=np.uint8)
WHITE = np.array([255, 255, 255], dtype=np.uint8)

base = np.array(Image.open(FIRST).convert("RGB"))
H, W = base.shape[:2]

# --- measure the example (top row) to derive the transformation parameters ---
green_mask = (np.abs(base.astype(int) - GREEN).sum(2) < 60).astype(np.uint8)
n, lab, stats, cents = cv2.connectedComponentsWithStats(green_mask)
comps = [(stats[i], cents[i]) for i in range(1, n)]
top = sorted([c for c in comps if c[1][1] < H / 2], key=lambda c: c[1][0])  # left→right
bottom = [c for c in comps if c[1][1] >= H / 2]
big, small, outline = top
SCALE = small[0][cv2.CC_STAT_WIDTH] / big[0][cv2.CC_STAT_WIDTH]            # ≈0.845
STROKE = (outline[0][cv2.CC_STAT_WIDTH] - small[0][cv2.CC_STAT_WIDTH]) / 2  # ≈3 px each side of edge
HALF = max(1.0, float(STROKE))

# --- the shape to animate: the bottom-row heart, as a polygon ---
hs = bottom[0][0]
x0, y0, w, h = hs[cv2.CC_STAT_LEFT], hs[cv2.CC_STAT_TOP], hs[cv2.CC_STAT_WIDTH], hs[cv2.CC_STAT_HEIGHT]
heart_id = [i for i in range(1, n) if np.allclose(cents[i], bottom[0][1])][0]
sub = (lab[y0:y0 + h, x0:x0 + w] == heart_id).astype(np.uint8)
cs, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
poly = cv2.approxPolyDP(max(cs, key=cv2.contourArea), 2, True).reshape(-1, 2).astype(float) + [x0, y0]
cx, cy = (x0 + (w - 1) / 2), (y0 + (h - 1) / 2)

# canvas with the heart erased (region around it is plain white)
clean = base.copy()
clean[lab == heart_id] = WHITE


def shape_mask(scale, outline_t):
    """Binary mask of the heart at given scale; outline_t in [0,1] hollows it out."""
    pts = (poly - [cx, cy]) * scale + [cx, cy]
    fill = np.zeros((H, W), np.uint8)
    cv2.fillPoly(fill, [np.round(pts).astype(np.int32)], 1)
    if outline_t <= 0:
        return fill.astype(bool)
    d_in = cv2.distanceTransform(fill, cv2.DIST_L2, 5)             # distance to edge inside
    d_out = cv2.distanceTransform(1 - fill, cv2.DIST_L2, 5)        # distance to edge outside
    max_in = d_in.max()
    inner_keep = HALF + (1 - outline_t) * (max_in - HALF)          # interior fill recedes to the border
    outer_grow = HALF * outline_t                                  # stroke grows outward
    return ((fill == 1) & (d_in <= inner_keep)) | ((fill == 0) & (d_out <= outer_grow))


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)


frames = []
n_scale = 8  # frames 0..8: scaling; frames 8..15: outlining
for i in range(N_FRAMES):
    if i == 0:
        frames.append(base.copy()); continue
    if i <= n_scale:
        s = 1 + (SCALE - 1) * ease(i / n_scale); o = 0.0
    else:
        s = SCALE; o = ease((i - n_scale) / (N_FRAMES - 1 - n_scale))
    img = clean.copy()
    img[shape_mask(s, o)] = GREEN
    frames.append(img)

os.makedirs(OUT_DIR, exist_ok=True)
tmp = os.path.join(OUT_DIR, "_frames")
shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
for i, f in enumerate(frames):
    Image.fromarray(f).save(os.path.join(tmp, f"{i:03d}.png"))
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(tmp, "%03d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT], check=True)
shutil.rmtree(tmp)
print("wrote", OUT, f"scale={SCALE:.3f} stroke_half={HALF}")
