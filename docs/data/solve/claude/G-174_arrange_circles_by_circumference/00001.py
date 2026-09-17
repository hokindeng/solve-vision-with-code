#!/usr/bin/env python3
"""Rearrange the 5 circles of first_frame.png into a centered horizontal row,
sorted left-to-right by circumference (largest -> smallest), and render the
motion as an 80-frame, 16 fps H.264 video."""
import os, subprocess, tempfile
import numpy as np
import cv2
from PIL import Image

APP = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 80
W = H = 1024
GAP = 30          # spacing between neighbouring circles in the final row
HOLD_START, HOLD_END = 6, 8   # frames held still at start / end

img = np.array(Image.open(SRC).convert("RGB"))
bg_color = np.array([255, 255, 255], np.uint8)
mask = (img != bg_color).any(axis=2).astype(np.uint8)
n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)

circles = []
for i in range(1, n):
    x, y, w, h, a = stats[i]
    if a < 200:
        continue
    m = (lab[y:y + h, x:x + w] == i)
    sprite = img[y:y + h, x:x + w].copy()
    circles.append(dict(w=w, h=h, sprite=sprite, mask=m,
                        cx=x + w / 2.0, cy=y + h / 2.0, d=max(w, h)))

# Circumference is proportional to diameter -> sort by diameter descending.
circles.sort(key=lambda c: -c["d"])
total = sum(c["d"] for c in circles) + GAP * (len(circles) - 1)
x0 = W / 2.0 - total / 2.0
for c in circles:
    c["tx"] = x0 + c["d"] / 2.0
    c["ty"] = H / 2.0
    x0 += c["d"] + GAP

background = np.full_like(img, 255)  # everything outside circles is pure white

def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)

def render(t):
    frame = background.copy()
    for c in circles:  # largest drawn first so smaller ones sit on top if overlapping mid-flight
        cx = c["cx"] + (c["tx"] - c["cx"]) * t
        cy = c["cy"] + (c["ty"] - c["cy"]) * t
        x = int(round(cx - c["w"] / 2.0)); y = int(round(cy - c["h"] / 2.0))
        region = frame[y:y + c["h"], x:x + c["w"]]
        region[c["mask"]] = c["sprite"][c["mask"]]
    return frame

os.makedirs(OUT_DIR, exist_ok=True)
move = N_FRAMES - HOLD_START - HOLD_END
with tempfile.TemporaryDirectory() as td:
    for f in range(N_FRAMES):
        if f == 0:
            frame = img
        else:
            t = np.clip((f - HOLD_START) / (move - 1), 0.0, 1.0)
            frame = render(ease(t))
        Image.fromarray(frame).save(os.path.join(td, f"{f:04d}.png"))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", os.path.join(td, "%04d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT], check=True)
print("wrote", OUT)
