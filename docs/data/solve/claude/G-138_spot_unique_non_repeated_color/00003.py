#!/usr/bin/env python3
"""Find the shape with the unique color and progressively outline it in black."""
import subprocess, numpy as np, cv2
from PIL import Image

SRC, OUT = "/app/first_frame.png", "/app/output/video.mp4"
N_FRAMES, FPS, THICK = 21, 16, 6

img = np.array(Image.open(SRC).convert("RGB"))
H, W = img.shape[:2]

# --- 1. Detect distinct fill colors (ignore background = most common color) ---
flat = img.reshape(-1, 3)
cols, counts = np.unique(flat, axis=0, return_counts=True)
bg = cols[counts.argmax()]
fills = [tuple(c) for c, n in zip(cols, counts) if n > 500 and not np.array_equal(c, bg)]

# --- 2. Count shapes (connected components) per color, pick the unique one ---
def mask_for(col):
    d = np.abs(img.astype(int) - np.array(col)).sum(axis=2)
    return (d < 40).astype(np.uint8)

shape_counts = {}
for col in fills:
    n, _ = cv2.connectedComponents(mask_for(col))
    shape_counts[col] = n - 1
unique = [c for c, n in shape_counts.items() if n == 1]
assert len(unique) == 1, shape_counts
target = unique[0]

# --- 3. Outer contour of that shape (include anti-aliased border pixels) ---
palette = np.array([tuple(bg)] + fills, dtype=int)          # index 0 = background
dists = np.stack([np.abs(img.astype(int) - c).sum(axis=2) for c in palette])
nearest = dists.argmin(axis=0)
m = (nearest == 1 + fills.index(target)).astype(np.uint8)   # pixels nearest to target color
n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
m = (lab == 1 + stats[1:, cv2.CC_STAT_AREA].argmax()).astype(np.uint8)
cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2)
# start drawing at the top-most-left point, go around once
start = np.lexsort((cnt[:, 0], cnt[:, 1]))[0]
cnt = np.roll(cnt, -start, axis=0)
seg = np.linalg.norm(np.diff(np.vstack([cnt, cnt[:1]]), axis=0), axis=1)
cum = np.concatenate([[0], np.cumsum(seg)])
total = cum[-1]

# --- 4. Render frames: outline grows from 0 to full length ---
frames = []
for i in range(N_FRAMES):
    f = img.copy()
    t = i / (N_FRAMES - 1)
    if t > 0:
        L = t * total
        k = int(np.searchsorted(cum, L, side="right"))
        pts = cnt[: min(k + 1, len(cnt))]
        if t >= 1.0:
            pts = np.vstack([cnt, cnt[:1]])
        if len(pts) >= 2:
            cv2.polylines(f, [pts.reshape(-1, 1, 2).astype(np.int32)], False,
                          (0, 0, 0), THICK, lineType=cv2.LINE_AA)
    frames.append(f)

# --- 5. Encode with ffmpeg (H.264, yuv420p) ---
p = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
     "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
     "-crf", "12", "-pix_fmt", "yuv420p", "-movflags", "+faststart", OUT],
    stdin=subprocess.PIPE)
for f in frames:
    p.stdin.write(f.tobytes())
p.stdin.close(); p.wait()
print("unique color", target, "shape counts", shape_counts, "->", OUT)
