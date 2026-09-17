#!/usr/bin/env python3
"""Sort scattered animal faces by size (largest -> smallest) and line them up on the baseline."""
import subprocess, os
import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 40

img = np.array(Image.open(SRC).convert("RGB"))
H, W, _ = img.shape
BG = img[0, 0].astype(int)

# --- detect baseline (long thin horizontal non-background line) ---
nonbg = np.abs(img.astype(int) - BG).sum(-1) > 0
row_counts = nonbg.sum(1)
line_rows = [y for y in range(H) if row_counts[y] > W * 0.5]
line_top, line_bot = min(line_rows), max(line_rows)
line_xs = np.where(nonbg[line_top])[0]
line_x0, line_x1 = int(line_xs.min()), int(line_xs.max())

# --- detect faces as connected components (excluding baseline) ---
mask = nonbg.copy()
mask[line_top - 2:line_bot + 3, :] = False
lab, n = ndimage.label(ndimage.binary_dilation(mask, iterations=4))
faces = []
for i, sl in enumerate(ndimage.find_objects(lab), 1):
    comp = (lab == i) & mask
    comp = ndimage.binary_fill_holes(comp)
    ys, xs = np.where(comp)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sprite = img[y0:y1, x0:x1].copy()
    alpha = comp[y0:y1, x0:x1]
    faces.append(dict(x=x0, y=y0, w=x1 - x0, h=y1 - y0, area=int(alpha.sum()),
                      sprite=sprite, alpha=alpha))

# background with faces removed
bg_img = img.copy()
for f in faces:
    sub = bg_img[f["y"]:f["y"] + f["h"], f["x"]:f["x"] + f["w"]]
    sub[f["alpha"]] = BG

# --- targets: sorted by size, largest first, left -> right, bottoms on baseline ---
order = sorted(range(len(faces)), key=lambda i: -faces[i]["area"])
total_w = sum(faces[i]["w"] for i in order)
span = line_x1 - line_x0 + 1
gap = (span - total_w) / (len(order) + 1)
cx = line_x0 + gap
for i in order:
    f = faces[i]
    f["tx"] = int(round(cx))
    f["ty"] = line_top - f["h"]          # bottom edge sits right on the baseline
    cx += f["w"] + gap

def ease(t):  # smooth ease-in-out
    return t * t * (3 - 2 * t)

def render(k):
    t = ease(k / (N_FRAMES - 1))
    frame = bg_img.copy()
    # draw largest last so it stays on top if paths cross
    for i in sorted(range(len(faces)), key=lambda i: faces[i]["area"]):
        f = faces[i]
        x = int(round(f["x"] + (f["tx"] - f["x"]) * t))
        y = int(round(f["y"] + (f["ty"] - f["y"]) * t))
        region = frame[y:y + f["h"], x:x + f["w"]]
        region[f["alpha"]] = f["sprite"][f["alpha"]]
    return frame

os.makedirs(OUT_DIR, exist_ok=True)
proc = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
     "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT],
    stdin=subprocess.PIPE)
for k in range(N_FRAMES):
    proc.stdin.write(render(k).tobytes())
proc.stdin.close()
proc.wait()
Image.fromarray(render(N_FRAMES - 1)).save(os.path.join(OUT_DIR, "last_frame.png"))
print("wrote", OUT, "order (largest->smallest):", [faces[i]["area"] for i in order])
