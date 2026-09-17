#!/usr/bin/env python3
"""Symbol deletion: fade out the red-bordered symbol, then slide the
remaining symbols left to close the gap. Every other pixel stays as in
first_frame.png."""
import subprocess, numpy as np
from PIL import Image

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS, N = 16, 46

img = np.array(Image.open(SRC).convert("RGB")).astype(np.float32)
H, W, _ = img.shape
bg = np.full_like(img, 255.0)  # background is uniform white

# --- locate cells: runs of non-white columns ---
nonwhite = (img != 255).any(axis=2)
cols = nonwhite.any(axis=0)
runs, s = [], None
for x in range(W + 1):
    on = x < W and cols[x]
    if on and s is None: s = x
    if not on and s is not None: runs.append((s, x - 1)); s = None

cells = []
for x0, x1 in runs:
    ys = np.where(nonwhite[:, x0:x1 + 1].any(axis=1))[0]
    y0, y1 = ys.min(), ys.max()
    patch = img[y0:y1 + 1, x0:x1 + 1]
    red = ((patch[..., 0] > 200) & (patch[..., 1] < 60) & (patch[..., 2] < 60)).sum()
    cells.append(dict(x0=x0, x1=x1, y0=y0, y1=y1, patch=patch, red=red))

# target = cell with the most pure-red pixels (the red border)
t = int(np.argmax([c["red"] for c in cells]))
# gap between neighbouring cell boxes (measured from non-target cells)
# pitch = distance between adjacent cell boxes, measured on two neighbouring
# non-target cells (the target's red border makes its box wider)
pitch = None
for i in range(len(cells) - 1):
    if i != t and i + 1 != t:
        pitch = cells[i + 1]["x0"] - cells[i]["x0"]; break
if pitch is None:  # only two cells: use box width + white gap around target
    o = [c for i, c in enumerate(cells) if i != t][0]
    pitch = (o["x1"] - o["x0"] + 1) + 10
shift = pitch  # amount the right-hand cells move left

def ease(u):  # smoothstep
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)

def paste(canvas, patch, x, y, alpha=1.0):
    """Alpha-blend patch onto canvas at integer (x, y); sub-pixel x via linear blend."""
    xi = int(np.floor(x)); fx = x - xi
    h, w = patch.shape[:2]
    for dx, wgt in ((xi, 1 - fx), (xi + 1, fx)):
        if wgt <= 1e-6: continue
        a = alpha * wgt
        # only blend where patch differs from white so overlaps stay clean
        region = canvas[y:y + h, dx:dx + w]
        region[:] = region * (1 - a) + patch * a

# timeline (frames): hold -> fade target -> hold -> slide -> hold
F_HOLD0, F_FADE_END, F_SLIDE_START, F_SLIDE_END = 4, 18, 22, 41

frames = []
for f in range(N):
    fade = ease((f - F_HOLD0) / (F_FADE_END - F_HOLD0))          # 0 -> 1
    slide = ease((f - F_SLIDE_START) / (F_SLIDE_END - F_SLIDE_START))
    if f == 0:
        frames.append(img.copy()); continue
    canvas = bg.copy()
    for i, c in enumerate(cells):
        if i == t:
            if fade < 1.0:
                paste(canvas, c["patch"], c["x0"], c["y0"], alpha=1.0 - fade)
        elif i > t:
            paste(canvas, c["patch"], c["x0"] - shift * slide, c["y0"])
        else:
            paste(canvas, c["patch"], c["x0"], c["y0"])
    frames.append(canvas)

frames = [np.clip(fr, 0, 255).astype(np.uint8) for fr in frames]

cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
       "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "medium", OUT]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for fr in frames:
    p.stdin.write(fr.tobytes())
p.stdin.close(); p.wait()
print(f"wrote {OUT}: {len(frames)} frames @ {FPS} fps, target cell index {t}, shift {shift}px")
