#!/usr/bin/env python3
"""Complete the shape_cycle sequence: heptagon, star, triangle, heptagon, ? -> star.

Animation: the question mark fades out while the star (copied from slot 2)
scales up and fades in at slot 5. Every other pixel is left untouched.
"""
import subprocess, shutil, os
import numpy as np
from PIL import Image

SRC = "/app/first_frame.png"
OUT_DIR = "/app/output"
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N = 16, 25

base = np.array(Image.open(SRC).convert("RGB"))
H, W, _ = base.shape
bg = base[0, 0].astype(int)
mask = np.abs(base.astype(int) - bg).sum(2) > 0

# --- locate elements as column-separated components -----------------------
cols = np.where(mask.any(0))[0]
groups, start, prev = [], cols[0], cols[0]
for c in cols[1:]:
    if c != prev + 1:
        groups.append((start, prev)); start = c
    prev = c
groups.append((start, prev))
boxes = []
for x0, x1 in groups:
    rows = np.where(mask[:, x0:x1 + 1].any(1))[0]
    boxes.append((int(x0), int(rows[0]), int(x1), int(rows[-1])))
assert len(boxes) == 5, boxes

# Slot centres: evenly spaced, derived from the outer shapes.
cx_first = (boxes[0][0] + boxes[0][2]) / 2
cx_last = (boxes[-1][0] + boxes[-1][2]) / 2
slot_cx = np.linspace(cx_first, cx_last, 5)
cy = H / 2  # shapes are vertically centred on the canvas

# Pattern: cycle length 3 -> element 5 equals element 2 (the star).
src_idx = (5 - 1) % 3          # 1 -> second element
sx0, sy0, sx1, sy1 = boxes[src_idx]
patch = base[sy0:sy1 + 1, sx0:sx1 + 1].copy()
patch_mask = mask[sy0:sy1 + 1, sx0:sx1 + 1]
patch_rgba = np.dstack([patch, (patch_mask * 255).astype(np.uint8)])
patch_img = Image.fromarray(patch_rgba, "RGBA")
# offset of the patch relative to the slot centre
dx0, dy0 = sx0 - slot_cx[src_idx], sy0 - cy
pw, ph = patch_img.size

qx0, qy0, qx1, qy1 = boxes[4]
q_region = base[qy0:qy1 + 1, qx0:qx1 + 1].astype(float)
white = np.full_like(q_region, 255.0)

def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)

def make_frame(i):
    t = i / (N - 1)
    frame = base.copy()
    # question mark fades out over the first ~55% of the video
    a_q = ease(t / 0.55)
    frame[qy0:qy1 + 1, qx0:qx1 + 1] = np.round(
        q_region * (1 - a_q) + white * a_q).astype(np.uint8)
    # star scales/fades in from ~30% to 100%
    a_s = ease((t - 0.30) / 0.70)
    if a_s <= 0:
        return frame
    s = 0.25 + 0.75 * a_s
    if i == N - 1:
        img = patch_img
        px, py = int(round(slot_cx[4] + dx0)), int(round(cy + dy0))
    else:
        nw, nh = max(1, int(round(pw * s))), max(1, int(round(ph * s)))
        img = patch_img.resize((nw, nh), Image.LANCZOS)
        ccx, ccy = slot_cx[4] + dx0 + pw / 2, cy + dy0 + ph / 2
        px, py = int(round(ccx - nw / 2)), int(round(ccy - nh / 2))
    canvas = Image.fromarray(frame, "RGB")
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    layer.paste(img, (px, py), img)
    if a_s < 1:
        al = np.array(layer)[..., 3].astype(float) * a_s
        layer.putalpha(Image.fromarray(np.round(al).astype(np.uint8)))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), layer).convert("RGB")
    return np.array(canvas)

tmp = os.path.join(OUT_DIR, "_frames")
shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
for i in range(N):
    Image.fromarray(make_frame(i)).save(os.path.join(tmp, f"{i:03d}.png"))

subprocess.run([
    "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
    "-i", os.path.join(tmp, "%03d.png"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
    "-r", str(FPS), OUT], check=True)
shutil.rmtree(tmp, ignore_errors=True)
print("wrote", OUT)
