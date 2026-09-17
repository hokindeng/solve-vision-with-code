#!/usr/bin/env python3
"""Regenerate /app/output/video.mp4: circle the vertex of the largest interior angle."""
import os, subprocess, tempfile
import numpy as np, cv2
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT = os.path.join(HERE, "output", "video.mp4")
FPS, N_FRAMES = 16, 22

base = np.array(Image.open(FIRST).convert("RGB"))
H, W = base.shape[:2]

# --- find triangle vertices -------------------------------------------------
gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
mask = (gray < 128).astype(np.uint8) * 255
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = max(cnts, key=cv2.contourArea)
eps = 0.01 * cv2.arcLength(cnt, True)
poly = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2).astype(float)
while len(poly) > 3:  # tighten until exactly 3 corners
    eps *= 1.3
    poly = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2).astype(float)
assert len(poly) == 3, poly
# refine with sub-pixel corner positions of the outline
verts = poly

def angle_at(i):
    a, b, c = verts[i], verts[(i + 1) % 3], verts[(i + 2) % 3]
    u, v = b - a, c - a
    return np.degrees(np.arccos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))

angles = [angle_at(i) for i in range(3)]
k = int(np.argmax(angles))
cx, cy = verts[k]
print("vertices:", verts.tolist())
print("angles:", [round(a, 1) for a in angles], "-> largest at", (round(cx), round(cy)))

# --- animation --------------------------------------------------------------
RADIUS, THICK = 34, 5
RED = (220, 30, 30)
HOLD = 3                       # frames identical to first frame
SWEEP_END = N_FRAMES - 3       # arc completes here, then hold on result
start_deg = -90                # begin sweep at top

def frame(i):
    im = Image.fromarray(base.copy())
    if i < HOLD:
        return im
    t = min(1.0, (i - HOLD) / (SWEEP_END - HOLD))
    t = t * t * (3 - 2 * t)    # smoothstep for pacing
    sweep = 360 * t
    # draw supersampled for smooth arc
    S = 4
    big = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    box = [(cx - RADIUS) * S, (cy - RADIUS) * S, (cx + RADIUS) * S, (cy + RADIUS) * S]
    if sweep >= 359.9:
        d.ellipse(box, outline=RED + (255,), width=THICK * S)
    else:
        d.arc(box, start=start_deg, end=start_deg + sweep, fill=RED + (255,), width=THICK * S)
    layer = big.resize((W, H), Image.LANCZOS)
    im = im.convert("RGBA")
    im.alpha_composite(layer)
    return im.convert("RGB")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with tempfile.TemporaryDirectory() as td:
    for i in range(N_FRAMES):
        frame(i).save(os.path.join(td, f"f{i:03d}.png"))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(td, "f%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT,
    ], check=True)
print("wrote", OUT)
