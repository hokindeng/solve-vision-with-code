#!/usr/bin/env python3
"""Identify the single unique shape in first_frame.png and circle it in red, animated."""
import subprocess, numpy as np, cv2
from PIL import Image, ImageDraw

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS, N = 16, 60
W = H = 1024

base = Image.open(SRC).convert("RGB")
a = np.array(base)

# --- detect shapes: any non-background pixel (background = most common color)
cols, cnt = np.unique(a.reshape(-1, 3), axis=0, return_counts=True)
bg = cols[cnt.argmax()]
mask = (np.abs(a.astype(int) - bg.astype(int)).sum(2) > 30).astype(np.uint8)
n, lab, st, cen = cv2.connectedComponentsWithStats(mask)
shapes = []
for i in range(1, n):
    x, y, w, h, area = st[i]
    if area < 50:
        continue
    c, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    per = cv2.arcLength(c[0], True)
    circ = 4 * np.pi * area / (per ** 2)
    verts = len(cv2.approxPolyDP(c[0], 0.02 * per, True))
    color = a[lab == i].mean(0)
    shapes.append(dict(cx=cen[i][0], cy=cen[i][1], w=w, h=h, area=area,
                       feat=np.array([circ * 10, np.log(area), verts / 4, *(color / 64)])))

# --- outlier: the shape whose feature vector is farthest from the median of the others
F = np.array([s["feat"] for s in shapes])
dist = [np.linalg.norm(F[i] - np.median(np.delete(F, i, 0), 0)) for i in range(len(F))]
u = shapes[int(np.argmax(dist))]
cx, cy = u["cx"], u["cy"]
R = int(0.5 * np.hypot(u["w"], u["h"]) + 18)   # circle radius with margin
RED, LW = (220, 30, 30), 6

def ease(t):
    return t * t * (3 - 2 * t)

frames = []
T0, T1 = 12, 50          # circle drawing spans frames 12..50; before = identification pause, after = hold
for k in range(N):
    im = base.copy()
    d = ImageDraw.Draw(im)
    if k >= T0:
        t = ease(min(1.0, (k - T0) / (T1 - T0)))
        sweep = 360 * t
        if sweep > 0:
            box = [cx - R, cy - R, cx + R, cy + R]
            if sweep >= 359.9:
                d.ellipse(box, outline=RED, width=LW)
            else:
                d.arc(box, start=-90, end=-90 + sweep, fill=RED, width=LW)
    frames.append(np.array(im))

# --- encode H.264 yuv420p
p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                      "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                      "-crf", "16", "-preset", "medium", OUT], stdin=subprocess.PIPE)
for f in frames:
    p.stdin.write(f.tobytes())
p.stdin.close(); p.wait()
print(f"unique shape at ({cx:.0f},{cy:.0f}), radius {R}; wrote {OUT}")
