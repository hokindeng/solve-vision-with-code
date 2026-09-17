#!/usr/bin/env python3
"""Rearrange the shapes in first_frame.png into a single sorted horizontal line.

Groups by type (squares, then triangles), sorted small -> large left to right.
Shapes are moved as exact pixel sprites (integer translation) so their
appearance is unchanged; the background is untouched.
"""
import os, subprocess, shutil
import numpy as np
import cv2
from PIL import Image

APP = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
W = H = 1024
FPS = 16
N_FRAMES = 96
HOLD_START, HOLD_END = 8, 8

img = np.array(Image.open(SRC).convert("RGB"))
bg_color = np.array([235, 235, 235], dtype=np.uint8)
mask = (np.abs(img.astype(int) - bg_color).sum(2) > 0).astype(np.uint8)
n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

shapes = []
for i in range(1, n):
    x, y, w, h, a = stats[i]
    m = (lab[y:y + h, x:x + w] == i)
    fill = a / float(w * h)
    kind = "square" if fill > 0.85 else "triangle"
    shapes.append(dict(kind=kind, x=int(x), y=int(y), w=int(w), h=int(h),
                       sprite=img[y:y + h, x:x + w].copy(), mask=m))

# Background: original image with shapes removed (flat bg colour).
background = img.copy()
background[mask.astype(bool)] = bg_color

# Target order: group by type, then size ascending.
order = ["square", "triangle"]
shapes.sort(key=lambda s: (order.index(s["kind"]), s["w"] * s["h"]))
gap = 40
total = sum(s["w"] for s in shapes) + gap * (len(shapes) - 1)
cx = (W - total) // 2
for s in shapes:
    s["tx"] = cx
    s["ty"] = H // 2 - s["h"] // 2
    cx += s["w"] + gap


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)  # smoothstep


def render(frame_idx):
    move_frames = N_FRAMES - HOLD_START - HOLD_END
    t = (frame_idx - HOLD_START) / float(move_frames - 1)
    canvas = background.copy()
    # Draw larger shapes first so smaller ones stay visible if paths cross.
    for s in sorted(shapes, key=lambda s: -s["w"] * s["h"]):
        e = ease(t)
        px = int(round(s["x"] + (s["tx"] - s["x"]) * e))
        py = int(round(s["y"] + (s["ty"] - s["y"]) * e))
        region = canvas[py:py + s["h"], px:px + s["w"]]
        region[s["mask"]] = s["sprite"][s["mask"]]
    return canvas


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = os.path.join(OUT_DIR, "_frames")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    for f in range(N_FRAMES):
        fr = img if f < HOLD_START else render(f)
        Image.fromarray(fr).save(os.path.join(tmp, f"{f:04d}.png"))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", os.path.join(tmp, "%04d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT], check=True)
    shutil.rmtree(tmp)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
