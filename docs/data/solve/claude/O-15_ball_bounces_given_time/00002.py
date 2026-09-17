#!/usr/bin/env python3
"""Regenerate /app/output/video.mp4: ball bounces 3 times elastically off the walls."""
import os, subprocess
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 80
N_BOUNCES = 3

WALL = np.array([100, 100, 100]); BALL = np.array([255, 192, 203]); ARROW = np.array([255, 140, 0])

first = np.array(Image.open(FIRST).convert("RGB"))
H, W, _ = first.shape

def mask_of(col):
    return (first == col).all(-1)

# --- measure scene -----------------------------------------------------------
wall = mask_of(WALL)
ys, xs = np.nonzero(wall)
x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
# inner edge of the wall: walk inward along the middle row/col until leaving gray
mid = H // 2
row = wall[mid]
inner_left = x0
while row[inner_left + 1]: inner_left += 1
inner_right = x1
while row[inner_right - 1]: inner_right -= 1
col = wall[:, W // 2]
inner_top = y0
while col[inner_top + 1]: inner_top += 1
inner_bot = y1
while col[inner_bot - 1]: inner_bot -= 1
# playable area for the ball (first empty pixel .. last empty pixel)
XMIN, XMAX = inner_left + 1, inner_right - 1
YMIN, YMAX = inner_top + 1, inner_bot - 1

ball = mask_of(BALL)
ys, xs = np.nonzero(ball)
cx0 = (xs.min() + xs.max()) / 2.0
cy0 = (ys.min() + ys.max()) / 2.0
R = (xs.max() - xs.min()) / 2.0  # ~30

arrow = mask_of(ARROW)
ys, xs = np.nonzero(arrow)
# direction = from ball centre towards the arrow tip (farthest orange pixel)
d = np.hypot(xs - cx0, ys - cy0)
tip = np.array([xs[d.argmax()], ys[d.argmax()]], float)
# refine with principal axis of the arrow pixels, oriented towards the tip
pts = np.stack([xs, ys], 1).astype(float)
pts_c = pts - pts.mean(0)
_, _, vt = np.linalg.svd(pts_c, full_matrices=False)
v = vt[0]
if np.dot(v, tip - np.array([cx0, cy0])) < 0:
    v = -v
v /= np.linalg.norm(v)

# --- simulate: straight segments between wall hits -----------------------------
lo = np.array([XMIN + R, YMIN + R]); hi = np.array([XMAX - R, YMAX - R])
pos = np.array([cx0, cy0]); vel = v.copy()
waypoints = [pos.copy()]
for _ in range(N_BOUNCES):
    ts = []
    for k in range(2):
        if vel[k] > 1e-12: ts.append(((hi[k] - pos[k]) / vel[k], k))
        elif vel[k] < -1e-12: ts.append(((lo[k] - pos[k]) / vel[k], k))
    t, k = min(ts)
    pos = pos + vel * t
    pos[k] = hi[k] if vel[k] > 0 else lo[k]  # snap exactly onto the wall contact
    vel[k] = -vel[k]                          # elastic reflection
    waypoints.append(pos.copy())

# arc-length parametrisation -> constant speed over the whole clip
seg = [np.linalg.norm(waypoints[i + 1] - waypoints[i]) for i in range(N_BOUNCES)]
cum = np.concatenate([[0], np.cumsum(seg)])
total = cum[-1]

def position(s):
    s = min(max(s, 0.0), total)
    i = min(np.searchsorted(cum, s, side="right") - 1, N_BOUNCES - 1)
    f = (s - cum[i]) / seg[i]
    return waypoints[i] * (1 - f) + waypoints[i + 1] * f

# --- render -------------------------------------------------------------------
background = first.copy()
background[ball] = 255  # the ball moves; everything else (walls, arrow) stays

def render(c):
    # PIL's ellipse rasteriser reproduces the original ball pixel-for-pixel
    im = Image.fromarray(background)
    ImageDraw.Draw(im).ellipse([c[0] - R, c[1] - R, c[0] + R, c[1] + R], fill=tuple(BALL))
    fr = np.array(im)
    fr[arrow] = ARROW  # arrow was drawn on top of the ball in the first frame
    return fr

os.makedirs(OUT_DIR, exist_ok=True)
ff = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
     "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "medium", OUT],
    stdin=subprocess.PIPE)
for i in range(N_FRAMES):
    if i == 0:
        fr = first
    else:
        fr = render(position(total * i / (N_FRAMES - 1)))
    ff.stdin.write(np.ascontiguousarray(fr, dtype=np.uint8).tobytes())
ff.stdin.close(); ff.wait()
if ff.returncode:
    raise SystemExit("ffmpeg failed")
print("wrote", OUT, "waypoints:", [tuple(np.round(w, 1)) for w in waypoints],
      "dir:", np.round(v, 4))
