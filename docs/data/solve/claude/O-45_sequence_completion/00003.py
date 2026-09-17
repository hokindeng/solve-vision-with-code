#!/usr/bin/env python3
"""Complete the color_cycle sequence: the '?' becomes the next element (green circle)."""
import os, shutil, subprocess
import numpy as np
from PIL import Image, ImageDraw

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 25, 16

base = np.array(Image.open(FIRST).convert("RGB"))
H, W, _ = base.shape

# --- analyse the scene ------------------------------------------------------
nonwhite = base.astype(int).sum(2) < 750
cols = nonwhite.any(0)
segs, start = [], None
for x in range(W):
    if cols[x] and start is None:
        start = x
    if not cols[x] and start is not None:
        segs.append((start, x - 1)); start = None

elems = []
for s, e in segs:
    yy = np.where(nonwhite[:, s:e + 1].any(1))[0]
    cx, cy = (s + e) / 2.0, (yy.min() + yy.max()) / 2.0
    elems.append(dict(x0=s, x1=e, y0=yy.min(), y1=yy.max(), cx=cx, cy=cy,
                      fill=tuple(int(v) for v in base[int(round(cy)), int(round(cx))])))

circles, qmark = elems[:-1], elems[-1]
radius = np.mean([(c["x1"] - c["x0"] + 1) / 2.0 for c in circles]) - 0.5  # PIL bbox is inclusive
pitch = np.mean(np.diff([c["cx"] for c in circles]))
cy = circles[0]["cy"]
target_cx = circles[-1]["cx"] + pitch  # should coincide with the '?'

# outline width: count black pixels from left edge along the centre row
c0 = circles[1]
row = base[int(round(c0["cy"])), c0["x0"]:c0["x0"] + 12].astype(int)
outline_w = int(np.argmax(row.sum(1) > 100))
outline_w = max(outline_w, 1)

# cycle detection: smallest period consistent with all colours
fills = [c["fill"] for c in circles]
period = next(p for p in range(1, len(fills) + 1)
              if all(fills[i] == fills[i % p] for i in range(len(fills))))
answer_fill = fills[len(fills) % period]

# --- rendering ---------------------------------------------------------------
# background with the '?' removed (white patch on its bbox, all white around it)
bg_clean = base.copy()
bg_clean[qmark["y0"]:qmark["y1"] + 1, qmark["x0"]:qmark["x1"] + 1] = 255

def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)

def render(i):
    t = i / (N_FRAMES - 1)
    # phase 1: '?' fades out (frames 0..~40%); phase 2: circle grows (~25%..100%)
    q_alpha = 1.0 - ease(min(1.0, t / 0.4))
    grow = ease(max(0.0, (t - 0.25) / 0.75))
    frame = (bg_clean.astype(float) * (1 - q_alpha) + base.astype(float) * q_alpha)
    frame = frame.round().clip(0, 255).astype(np.uint8)
    if grow > 0:
        r = radius * grow
        img = Image.fromarray(frame)
        d = ImageDraw.Draw(img)
        w = max(1, int(round(outline_w * grow))) if r > outline_w else 1
        d.ellipse([target_cx - r, cy - r, target_cx + r, cy + r],
                  fill=answer_fill, outline=(0, 0, 0), width=w)
        frame = np.array(img)
    if i == 0:
        frame = base.copy()
    return frame

os.makedirs(OUT_DIR, exist_ok=True)
frames_dir = os.path.join(OUT_DIR, "frames")
os.makedirs(frames_dir, exist_ok=True)
for i in range(N_FRAMES):
    Image.fromarray(render(i)).save(os.path.join(frames_dir, f"f{i:03d}.png"))

subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                "-i", os.path.join(frames_dir, "f%03d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
                "-preset", "slow", OUT], check=True)
shutil.rmtree(frames_dir, ignore_errors=True)
print(f"period={period} answer={answer_fill} radius={radius:.1f} pitch={pitch:.2f} "
      f"target=({target_cx:.1f},{cy:.1f}) qmark=({qmark['cx']:.1f},{qmark['cy']:.1f}) -> {OUT}")
