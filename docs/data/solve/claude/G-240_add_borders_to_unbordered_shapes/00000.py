#!/usr/bin/env python3
"""Add a thin black border to every unbordered shape, animated as a sweep."""
import os, subprocess, numpy as np, cv2
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 80, 16
BORDER = 6          # thickness of the existing border (measured ~6 px)
HOLD_END = 6        # frames held on the finished result

img = np.array(Image.open(SRC).convert("RGB"))
H, W, _ = img.shape
bg = np.array([255, 255, 255], np.uint8)
black = np.array([0, 0, 0], np.uint8)

# --- find shapes: connected components of non-background, non-black pixels
non_bg = ~(img == bg).all(2)
is_black = (img == black).all(2)
fill = (non_bg & ~is_black).astype(np.uint8)
n, lab, st, cen = cv2.connectedComponentsWithStats(fill, 8)

shapes = []
for i in range(1, n):
    if st[i, cv2.CC_STAT_AREA] < 200:
        continue
    m = (lab == i).astype(np.uint8)
    ring = cv2.dilate(m, np.ones((3, 3), np.uint8)) - m   # 1px outside ring
    frac_black = is_black[ring.astype(bool)].mean()
    if frac_black > 0.5:
        continue  # already bordered
    r = BORDER // 2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    outer = cv2.dilate(m, k).astype(bool)
    inner = cv2.erode(m, k).astype(bool)
    band = outer & ~inner
    # don't paint over other shapes or existing black borders
    band &= ~(non_bg & ~m.astype(bool))
    shapes.append((band, cen[i]))

# order shapes top-left to bottom-right (reading order)
shapes.sort(key=lambda s: (round(s[1][1] / 200), s[1][0]))

ys, xs = np.mgrid[0:H, 0:W]
angle_maps = []
for band, (cx, cy) in shapes:
    ang = (np.arctan2(ys - cy, xs - cx) + np.pi / 2) % (2 * np.pi) / (2 * np.pi)
    angle_maps.append(ang)

# timing: staggered sweeps, each covering a window of the active period
active = N_FRAMES - 1 - HOLD_END
k = len(shapes)
sweep_len = active * 0.55
starts = [1 + (active - sweep_len) * i / max(k - 1, 1) for i in range(k)]

def frame_at(f):
    out = img.copy()
    for (band, _), ang, s in zip(shapes, angle_maps, starts):
        t = (f - s) / sweep_len
        if t <= 0:
            continue
        t = min(t, 1.0)
        t = t * t * (3 - 2 * t)          # ease in-out
        sel = band & (ang <= t)
        out[sel] = black
    return out

os.makedirs(OUT_DIR, exist_ok=True)
proc = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
     "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT],
    stdin=subprocess.PIPE)
for f in range(N_FRAMES):
    proc.stdin.write(frame_at(f).tobytes())
proc.stdin.close(); proc.wait()
print("wrote", OUT, "shapes bordered:", k)
