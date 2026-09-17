#!/usr/bin/env python3
"""Find the single pentagon in first_frame.png and mark it with a red circle
that expands from its centre until it encircles the shape."""
import os, subprocess, numpy as np, cv2
from PIL import Image, ImageDraw

APP = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 30, 16

base = np.array(Image.open(SRC).convert("RGB"))
H, W = base.shape[:2]

# --- detect shapes: anything that is not the (white) background -------------
bg = np.median(base.reshape(-1, 3), axis=0)
mask = (np.abs(base.astype(int) - bg).sum(axis=2) > 40).astype(np.uint8) * 255
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

def vertex_count(c):
    peri = cv2.arcLength(c, True)
    return len(cv2.approxPolyDP(c, 0.03 * peri, True))

shapes = [(c, vertex_count(c)) for c in contours if cv2.contourArea(c) > 200]
pentas = [c for c, n in shapes if n == 5]
assert len(pentas) == 1, f"expected exactly one pentagon, found {len(pentas)} ({[n for _, n in shapes]})"
pent = pentas[0]

# centre + radius that fully encloses the pentagon
(cx, cy), r_shape = cv2.minEnclosingCircle(pent)
R_FINAL = r_shape + 14          # small margin outside the shape
THICK = 6

# --- render frames ----------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
frames = []
for i in range(N_FRAMES):
    img = Image.fromarray(base.copy())
    if i > 0:
        t = i / (N_FRAMES - 1)
        t = 1 - (1 - t) ** 2          # ease-out: fast start, settles at end
        r = R_FINAL * t
        if r > THICK / 2:
            d = ImageDraw.Draw(img)
            d.ellipse([cx - r, cy - r, cx + r, cy + r],
                      outline=(255, 0, 0), width=THICK)
    frames.append(np.array(img))

# --- encode -----------------------------------------------------------------
tmp = os.path.join(OUT_DIR, "_frames")
os.makedirs(tmp, exist_ok=True)
for i, f in enumerate(frames):
    Image.fromarray(f).save(os.path.join(tmp, f"{i:04d}.png"))
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                "-i", os.path.join(tmp, "%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
                "-preset", "slow", OUT], check=True)
for fn in os.listdir(tmp):
    os.remove(os.path.join(tmp, fn))
os.rmdir(tmp)
print(f"pentagon at ({cx:.0f},{cy:.0f}), final radius {R_FINAL:.0f}; wrote {OUT}")
