#!/usr/bin/env python3
"""Animate the ball rolling along the platform path in first_frame.png."""
import math
import subprocess
import numpy as np
import cv2
from PIL import Image

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
N_FRAMES, FPS, SS = 64, 16, 4  # SS = supersampling for anti-aliased ball

img = np.array(Image.open(SRC).convert("RGB"))
H, W = img.shape[:2]
r, g, b = [img[..., i].astype(int) for i in range(3)]

# --- ball geometry (pink fill + dark outline) ---------------------------------
ball_mask = ((r > 150) & (g < 100) & (b > 50))          # fill + outline
ys, xs = np.nonzero(ball_mask)
cx0, cy0 = (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0
R = (xs.max() - xs.min() + 1) / 2.0
FILL = (255, 20, 147)
EDGE = (175, 0, 67)
# outline thickness: count edge-coloured pixels along the centre row
row = img[int(round(cy0)), xs.min():xs.max() + 1]
EDGE_T = max(1, int(np.sum(np.all(row == EDGE, axis=1)) / 2))

# --- platforms (teal squares), ordered along the path --------------------------
teal = (g > 120) & (r < 120) & (b > 120)
n, lab, stats, cents = cv2.connectedComponentsWithStats(teal.astype(np.uint8))
plats = [tuple(cents[i]) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 50]
# order by walking from the platform nearest the ball
order = []
cur = np.array([cx0, cy0])
rem = list(plats)
while rem:
    j = min(range(len(rem)), key=lambda k: np.hypot(*(np.array(rem[k]) - cur)))
    cur = np.array(rem.pop(j)); order.append(tuple(cur))
plats = np.array(order)
PLAT_HALF = 8.0

# rest position of the ball on each platform: offset along the upward normal
def normal_at(i):
    a = plats[max(i - 1, 0)]; c = plats[min(i + 1, len(plats) - 1)]
    t = c - a; t /= np.linalg.norm(t)
    nrm = np.array([-t[1], t[0]])
    if nrm[1] > 0: nrm = -nrm            # point upward (negative y)
    return nrm

keys = [np.array([cx0, cy0])] + [plats[i] + normal_at(i) * (R + PLAT_HALF - 1) for i in range(len(plats))]
keys = np.array(keys)

# --- background with the ball removed -----------------------------------------
bg = img.copy()
bg[ball_mask] = (255, 255, 255)
ball_mask_d = cv2.dilate(ball_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
bg[ball_mask_d & (img.sum(-1) > 600)] = (255, 255, 255)   # anti-aliased fringe

def catmull_rom(pts, s):
    """Position on Catmull-Rom spline through pts at parameter s in [0, len-1]."""
    m = len(pts) - 1
    s = min(max(s, 0.0), m - 1e-9)
    i = int(math.floor(s)); t = s - i
    p0, p1, p2, p3 = pts[max(i - 1, 0)], pts[i], pts[min(i + 1, m)], pts[min(i + 2, m)]
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3)

def ball_pos(f):
    move_frames = 54                      # motion ends here, rest afterwards
    u = min(f / move_frames, 1.0)
    u = u * u * (3 - 2 * u)               # smoothstep: ease in / out
    s = u * (len(keys) - 1)
    p = catmull_rom(keys, s)
    seg_t = s - math.floor(s)
    hop = 9.0 * math.sin(math.pi * seg_t) if s < len(keys) - 1 else 0.0
    return p[0], p[1] - hop

def draw_ball(base, cx, cy):
    big = cv2.resize(base, (W * SS, H * SS), interpolation=cv2.INTER_NEAREST)
    c = (int(round(cx * SS)), int(round(cy * SS)))
    cv2.circle(big, c, int(round(R * SS)), EDGE[::-1][::-1], -1, lineType=cv2.LINE_AA)
    cv2.circle(big, c, int(round((R - EDGE_T) * SS)), FILL, -1, lineType=cv2.LINE_AA)
    return cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)

frames = []
for f in range(N_FRAMES):
    if f == 0:
        frames.append(img.copy())         # exact first frame
        continue
    cx, cy = ball_pos(f)
    frames.append(draw_ball(bg, cx, cy))

ff = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
     "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
     "-crf", "16", "-preset", "medium", OUT], stdin=subprocess.PIPE)
for fr in frames:
    ff.stdin.write(np.ascontiguousarray(fr).tobytes())
ff.stdin.close(); ff.wait()
print("wrote", OUT, "frames", len(frames))
