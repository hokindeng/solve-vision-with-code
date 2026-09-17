"""Add a thin black border to every unbordered shape in first_frame.png, animated over 80 frames."""
import os, subprocess, tempfile
import numpy as np, cv2
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT = os.path.join(HERE, "output", "video.mp4")
FPS, N_FRAMES, THICK = 16, 80, 4

img = np.array(Image.open(SRC).convert("RGB"))
H, W, _ = img.shape
bg = img[0, 0]

# Segment shapes: connected components of non-background pixels.
fg = (np.abs(img.astype(int) - bg.astype(int)).sum(axis=2) > 30).astype(np.uint8)
n, lab, stats, cent = cv2.connectedComponentsWithStats(fg, connectivity=8)

k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * THICK + 1, 2 * THICK + 1))
jobs = []  # (border_mask, centroid) for shapes lacking a border
for i in range(1, n):
    if stats[i, cv2.CC_STAT_AREA] < 200:
        continue
    m = (lab == i).astype(np.uint8)
    ring = m - cv2.erode(m, k)                       # inner rim of the shape
    rim_px = img[ring.astype(bool)]
    dark_frac = (rim_px.max(axis=1) < 60).mean()     # already has a black border?
    if dark_frac > 0.5:
        continue
    jobs.append((ring.astype(bool), cent[i]))

# Sort shapes left-to-right, top-to-bottom for a readable sequence.
jobs.sort(key=lambda j: (round(j[1][1] / 200), j[1][0]))

yy, xx = np.mgrid[0:H, 0:W]
def frame_at(t):
    """t in [0,1]: overall progress. Each shape is traced by an angular sweep in its own time slot."""
    out = img.copy()
    if not jobs:
        return out
    slot = 1.0 / len(jobs)
    for j, (ring, (cx, cy)) in enumerate(jobs):
        p = np.clip((t - j * slot) / (slot * 0.9), 0, 1)  # small pause between shapes
        if p <= 0:
            continue
        ang = (np.arctan2(yy - cy, xx - cx) + np.pi / 2) % (2 * np.pi)  # start at top, go clockwise
        sel = ring & (ang <= p * 2 * np.pi + 1e-6) if p < 1 else ring
        out[sel] = 0
    return out

with tempfile.TemporaryDirectory() as td:
    for f in range(N_FRAMES):
        t = 0.0 if f == 0 else min(1.0, f / (N_FRAMES - 4))  # complete a few frames before the end
        Image.fromarray(frame_at(t)).save(os.path.join(td, f"{f:04d}.png"))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", os.path.join(td, "%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "12", OUT], check=True)
print("wrote", OUT, "shapes bordered:", len(jobs))
