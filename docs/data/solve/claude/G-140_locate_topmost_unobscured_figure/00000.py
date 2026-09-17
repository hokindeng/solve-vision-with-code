#!/usr/bin/env python3
"""Outline the topmost (unobscured) shape with a red outline, animated over 40 frames."""
import subprocess, numpy as np, cv2
from PIL import Image, ImageDraw

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FRAMES, FPS = 40, 16
RED, WIDTH = (255, 0, 0), 6

base = np.array(Image.open(SRC).convert("RGB"))
H, W, _ = base.shape

# --- detect shapes: each distinct non-background flat colour is one shape ---
cols, counts = np.unique(base.reshape(-1, 3), axis=0, return_counts=True)
bg = cols[np.argmax(counts)]
shapes = []
for col, cnt in zip(cols, counts):
    if (col == bg).all() or cnt < 500:
        continue
    mask = (base == col).all(2).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_ = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(cnt_)
    # topmost = unobscured: visible region equals its convex hull (nothing bitten out)
    solidity = cv2.contourArea(cnt_) / max(cv2.contourArea(hull), 1)
    shapes.append((solidity, cv2.contourArea(cnt_), cnt_, col))
shapes.sort(key=lambda s: (-round(s[0], 3), -s[1]))
top = shapes[0][2]
poly = cv2.approxPolyDP(top, 3, True).reshape(-1, 2).astype(float)
print("topmost colour", shapes[0][3], "vertices", poly.astype(int).tolist())

# --- perimeter path, param t in [0,1] ---
pts = np.vstack([poly, poly[:1]])
seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
cum = np.concatenate([[0], np.cumsum(seg)])
total = cum[-1]

def point_at(d):
    i = np.searchsorted(cum, d, side="right") - 1
    i = min(i, len(seg) - 1)
    f = (d - cum[i]) / seg[i]
    return tuple(pts[i] + f * (pts[i + 1] - pts[i]))

def frame(k):
    img = Image.fromarray(base.copy())
    if k == 0:
        return img
    t = k / (FRAMES - 1)
    t = t * t * (3 - 2 * t)  # ease in/out
    d = t * total
    dr = ImageDraw.Draw(img)
    path = [tuple(pts[0])]
    for i in range(len(seg)):
        if cum[i + 1] <= d:
            path.append(tuple(pts[i + 1]))
        else:
            path.append(point_at(d)); break
    if k == FRAMES - 1:
        dr.polygon([tuple(p) for p in poly], outline=RED, width=WIDTH)
    else:
        dr.line(path, fill=RED, width=WIDTH, joint="curve")
        for p in path[:-1]:
            dr.ellipse([p[0]-WIDTH/2, p[1]-WIDTH/2, p[0]+WIDTH/2, p[1]+WIDTH/2], fill=RED)
    return img

ff = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
     "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT],
    stdin=subprocess.PIPE)
for k in range(FRAMES):
    ff.stdin.write(np.asarray(frame(k), dtype=np.uint8).tobytes())
ff.stdin.close(); ff.wait()
print("wrote", OUT)
