#!/usr/bin/env python3
"""Remove all cone objects (the triangles) from the scene by dissolving them
into the white background over the full clip; everything else is untouched."""
import subprocess, os
import numpy as np
from PIL import Image
import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "first_frame.png")
OUT_DIR = os.path.join(ROOT, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
W = H = 1024
FPS = 16
N_FRAMES = 96

first = np.array(Image.open(SRC).convert("RGB"))
bg = np.array([255, 255, 255], dtype=np.uint8)  # uniform white background

# --- find the cone objects: connected components that are triangles --------
fg = np.any(first != bg, axis=2).astype(np.uint8)
n, labels = cv2.connectedComponents(fg, connectivity=8)
cone_masks = []
for k in range(1, n):
    comp = (labels == k).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.03 * peri, True)
    if len(approx) == 3:  # a triangle = 2-D cone
        cone_masks.append(comp.astype(np.float32))
print(f"found {len(cone_masks)} cone(s) among {n - 1} objects")

# --- staggered dissolve so the action spans the whole 6 s -----------------
def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)

# cone i fades over [start_i, end_i] in frames; windows overlap and cover 0..N-1
windows = []
span = 60
for i, _ in enumerate(cone_masks):
    s = 0 if len(cone_masks) == 1 else round(i * (N_FRAMES - 1 - span) / (len(cone_masks) - 1))
    windows.append((s, s + span))

os.makedirs(OUT_DIR, exist_ok=True)
cmd = ["ffmpeg", "-y", "-loglevel", "error",
       "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
       "-movflags", "+faststart", OUT]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
base = first.astype(np.float32)
for f in range(N_FRAMES):
    frame = base.copy()
    for m, (s, e) in zip(cone_masks, windows):
        a = smoothstep((f - s) / (e - s))
        if f == N_FRAMES - 1:
            a = 1.0
        if a > 0:
            w = (m * a)[..., None]
            frame = frame * (1 - w) + bg.astype(np.float32) * w
    proc.stdin.write(np.clip(frame + 0.5, 0, 255).astype(np.uint8).tobytes())
proc.stdin.close()
proc.wait()
print("wrote", OUT)
