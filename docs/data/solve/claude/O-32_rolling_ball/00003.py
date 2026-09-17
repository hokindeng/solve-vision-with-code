#!/usr/bin/env python3
"""Animate the pink ball rolling along the dashed path in first_frame.png."""
import subprocess, os
import numpy as np
from PIL import Image
import cv2

APP = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N = 16, 64

img = np.array(Image.open(SRC).convert("RGB"))
H, W, _ = img.shape
a = img.astype(int)

def near(c, tol=60):
    return np.abs(a - np.array(c)).sum(-1) < tol

# ---- ball sprite -----------------------------------------------------------
ball_mask = near((255, 105, 180)) | near((175, 25, 100))
ys, xs = np.nonzero(ball_mask)
bx0, bx1, by0, by1 = xs.min(), xs.max(), ys.min(), ys.max()
ball_cx, ball_cy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
sprite = img[by0:by1 + 1, bx0:bx1 + 1].copy()
sprite_mask = ball_mask[by0:by1 + 1, bx0:bx1 + 1]
sh, sw = sprite.shape[:2]

# ---- dashed path -------------------------------------------------------------
path_mask = near((70, 130, 200)) | near((10, 70, 140))
n, lab, stats, cent = cv2.connectedComponentsWithStats(path_mask.astype(np.uint8))
comps = [(stats[i], cent[i]) for i in range(1, n)]
# full dashes only (largest area class) for template + spacing
areas = np.array([s[cv2.CC_STAT_AREA] for s, _ in comps])
full = [c for c, ar in zip(comps, areas) if ar > 0.9 * areas.max()]
# order by distance from the ball (far -> near)
full.sort(key=lambda c: -np.hypot(c[1][0] - ball_cx, c[1][1] - ball_cy))
centers = np.array([c[1] for c in full])
step = (centers[-1] - centers[0]) / (len(centers) - 1)        # vector per dash toward ball
far_end = centers[0]                                            # final platform
# template: one full dash
s, c = full[len(full) // 2]
x, y, w, h = s[cv2.CC_STAT_LEFT], s[cv2.CC_STAT_TOP], s[cv2.CC_STAT_WIDTH], s[cv2.CC_STAT_HEIGHT]
tmpl = img[y:y + h, x:x + w].copy()
tmpl_mask = path_mask[y:y + h, x:x + w] & (lab[y:y + h, x:x + w] == lab[int(c[1]), int(c[0])])
tmpl_off = (x - c[0], y - c[1])

# ---- background: remove ball, extrapolate hidden dashes ----------------------
bg = img.copy()
bg[ball_mask] = 255
k = len(centers) - 1
while True:
    k += 1
    cc = centers[0] + step * k
    # only dashes between the ball centre and the target exist (path starts at the ball)
    along = ((cc[0] - ball_cx) * -step[0] + (cc[1] - ball_cy) * -step[1]) / np.hypot(*step)
    if along < -0.5 * np.hypot(*step):
        break
    px, py = int(round(cc[0] + tmpl_off[0])), int(round(cc[1] + tmpl_off[1]))
    for dy in range(h):
        for dx in range(w):
            if tmpl_mask[dy, dx]:
                X, Y = px + dx, py + dy
                if 0 <= X < W and 0 <= Y < H and ball_mask[Y, X]:
                    bg[Y, X] = tmpl[dy, dx]

# ---- motion ----------------------------------------------------------------
start = np.array([ball_cx, ball_cy])
end = np.array(far_end, dtype=float)
d = end - start
L = np.hypot(*d)
u = d / L
perp = np.array([u[1], -u[0]])      # "up" side (screen y down)
if perp[1] > 0:
    perp = -perp

def ease(t):
    return 3 * t**2 - 2 * t**3

move_frames = N - 6                 # rest for the last frames
hops = 4                            # land on a few platforms along the way
def pos(i):
    t = min(1.0, i / (move_frames - 1))
    s_ = ease(t)
    p = start + d * s_
    # small arcs between landings; amplitude fades toward the final platform
    arc = abs(np.sin(np.pi * hops * s_))
    amp = 14.0 * (1 - 0.5 * s_)
    return p + perp * arc * amp

def paste(frame, cx, cy):
    x0 = int(round(cx - ball_cx)) + bx0
    y0 = int(round(cy - ball_cy)) + by0
    xs0, ys0 = max(0, x0), max(0, y0)
    xs1, ys1 = min(W, x0 + sw), min(H, y0 + sh)
    if xs1 <= xs0 or ys1 <= ys0:
        return
    sub = sprite[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
    m = sprite_mask[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
    frame[ys0:ys1, xs0:xs1][m] = sub[m]

os.makedirs(OUT_DIR, exist_ok=True)
ff = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
     "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT],
    stdin=subprocess.PIPE)
for i in range(N):
    if i == 0:
        frame = img.copy()
    else:
        frame = bg.copy()
        cx, cy = pos(i)
        paste(frame, cx, cy)
    ff.stdin.write(frame.tobytes())
ff.stdin.close()
ff.wait()
print("wrote", OUT)
