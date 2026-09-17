"""Outline all trapezoids inside the square region, animated over 40 frames."""
import subprocess, os
import numpy as np
import cv2
from PIL import Image

SRC = "/app/first_frame.png"
OUT_DIR = "/app/output"
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 40, 16
OUTLINE_COLOR = (0, 190, 60)     # green outline (RGB)
GAP, THICK = 3, 5                # px gap from the shape, ring thickness

base = np.array(Image.open(SRC).convert("RGB"))
H, W, _ = base.shape

# ---- 1. find the square region (thin black frame) --------------------------
black = np.all(base < 40, axis=2).astype(np.uint8)
cnts, _ = cv2.findContours(black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
square = None
for c in cnts:
    if cv2.contourArea(c) < 5000:
        continue
    approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
    x, y, w, h = cv2.boundingRect(c)
    if len(approx) == 4 and 0.9 < w / h < 1.1:
        square = (x, y, w, h)
if square is None:
    raise SystemExit("square region not found")
sx, sy, sw, sh = square
region = np.zeros((H, W), np.uint8)
region[sy + 3:sy + sh - 3, sx + 3:sx + sw - 3] = 1

# ---- 2. find shapes inside the region (anything non-white, incl. dark edge) --
nonwhite = (np.any(base < 245, axis=2) & (region == 1)).astype(np.uint8)
nonwhite = cv2.morphologyEx(nonwhite, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
n, labels = cv2.connectedComponents(nonwhite)

def is_trapezoid(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    approx = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True).reshape(-1, 2)
    if len(approx) != 4:
        return False
    # side direction vectors
    v = [approx[(i + 1) % 4] - approx[i] for i in range(4)]
    def ang(a, b):
        cosv = abs(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        return np.degrees(np.arccos(np.clip(cosv, 0, 1)))
    par02 = ang(v[0], v[2]) < 6
    par13 = ang(v[1], v[3]) < 6
    if par02 and par13:          # parallelogram / rectangle -> not a trapezoid
        return False
    return par02 or par13

targets = []
for lab in range(1, n):
    m = (labels == lab).astype(np.uint8)
    if m.sum() < 150:
        continue
    if is_trapezoid(m):
        targets.append(m)
# order left-to-right for a deterministic animation
targets.sort(key=lambda m: cv2.boundingRect(m)[0])
print(f"square at {square}, trapezoids found: {len(targets)}")

# ---- 3. build the outline ring for each trapezoid ---------------------------
def ring(mask):
    k_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * GAP + 1,) * 2)
    k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (GAP + THICK) + 1,) * 2)
    inner = cv2.dilate(mask, k_in)
    outer = cv2.dilate(mask, k_out)
    return ((outer - inner) > 0)

rings = []
for m in targets:
    r = ring(m)
    ys, xs = np.nonzero(m)
    cy, cx = ys.mean(), xs.mean()
    ry, rx = np.nonzero(r)
    theta = (np.arctan2(ry - cy, rx - cx) + np.pi / 2) % (2 * np.pi)  # start at top, clockwise
    rings.append((r, ry, rx, theta))

# ---- 4. animate: each outline is swept in clockwise, one after another -----
frames = []
start, end = 2, N_FRAMES - 3          # frames during which drawing happens
per = (end - start) / max(len(rings), 1)
for f in range(N_FRAMES):
    img = base.copy()
    for i, (r, ry, rx, theta) in enumerate(rings):
        t0 = start + i * per
        prog = np.clip((f - t0) / per, 0, 1)
        if prog <= 0:
            continue
        sel = theta <= prog * 2 * np.pi + 1e-6
        img[ry[sel], rx[sel]] = OUTLINE_COLOR
    frames.append(img)

# ---- 5. encode ------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
       "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for fr in frames:
    p.stdin.write(np.ascontiguousarray(fr).tobytes())
p.stdin.close()
p.wait()
print("wrote", OUT)
