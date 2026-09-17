#!/usr/bin/env python3
"""Regenerate /app/output/video.mp4: shift every teal block 3 grid cells downward."""
import os, subprocess
import numpy as np
from PIL import Image

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N, FPS, NFRAMES, SHIFT = 20, 16, 35, 3

first = np.array(Image.open(FIRST).convert("RGB"))
H, W, _ = first.shape
GRAY = np.array([51, 51, 51])
is_gray = (first == GRAY).all(axis=2)
# Grid lines detected from the image (rows/cols that are mostly gray), plus the outer edge.
lines = [int(y) for y in np.where(is_gray.mean(axis=1) > 0.5)[0]] + [H]
assert lines == [int(x) for x in np.where(is_gray.mean(axis=0) > 0.5)[0]] + [W]
assert len(lines) == N + 1, lines

TEAL = np.array([0, 128, 128])
is_teal = (first == TEAL).all(axis=2)

# Detect blocks: cells whose center pixel is teal.  Sprite = cell interior inset by 4px.
INSET, SIZE = 4, 44
blocks = []
for r in range(N):
    for c in range(N):
        y, x = lines[r], lines[c]
        cy, cx = y + (lines[r + 1] - y) // 2, x + (lines[c + 1] - x) // 2
        if is_teal[cy, cx]:
            sprite = first[y + INSET:y + INSET + SIZE, x + INSET:x + INSET + SIZE].copy()
            # sanity: sprite border is black, surroundings inside the cell are white
            assert (sprite[0] == 0).all() and (sprite[-1] == 0).all() and (sprite[:, 0] == 0).all() and (sprite[:, -1] == 0).all()
            assert (first[y + 1:y + INSET, x + 1:lines[c + 1]] == 255).all()
            assert (first[y + INSET + SIZE:lines[r + 1], x + 1:lines[c + 1]] == 255).all()
            blocks.append((r, c, sprite))
assert len(blocks) == 55, len(blocks)
assert all(r + SHIFT < N for r, _, _ in blocks)

# Background: first frame with blocks erased (cell interiors are white).
bg = first.copy()
for r, c, _ in blocks:
    y, x = lines[r], lines[c]
    bg[y + 1:lines[r + 1], x + 1:lines[c + 1]] = 255

def ease(t):  # smoothstep
    return t * t * (3 - 2 * t)

frames = []
for i in range(NFRAMES):
    t = ease(i / (NFRAMES - 1))
    frame = bg.copy()
    for r, c, sprite in blocks:
        y0 = lines[r] + INSET
        y1 = lines[r + SHIFT] + INSET
        y = int(round(y0 + (y1 - y0) * t))
        x = lines[c] + INSET
        frame[y:y + SIZE, x:x + SIZE] = sprite
    frames.append(frame)

frames[0] = first.copy()  # exact first frame
os.makedirs(OUT_DIR, exist_ok=True)
raw = np.stack(frames).tobytes()
subprocess.run([
    "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
    "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT,
], input=raw, check=True)
print("wrote", OUT, len(frames), "frames")
