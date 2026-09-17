#!/usr/bin/env python3
"""Find the rectangle closest to a square and circle it in red, animated over 3 s."""
import math
import subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS, N_FRAMES = 16, 48
W = H = 1024

base = np.array(Image.open(SRC).convert("RGB"))

# --- Step 1: detect the axis-aligned rectangles ---------------------------------
gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
mask = (gray < 250).astype(np.uint8)
n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
rects = []
for x, y, w, h, area in stats[1:]:
    if area < 200:
        continue
    rects.append((x, y, w, h))

# --- Step 2: compare aspect ratios; closest to 1:1 wins -------------------------
def squareness(r):
    _, _, w, h = r
    ratio = max(w, h) / min(w, h)   # >= 1, equals 1 for a perfect square
    return ratio

for r in sorted(rects, key=squareness):
    print(f"rect at ({r[0]},{r[1]}) size {r[2]}x{r[3]}  ratio={squareness(r):.3f}")
target = min(rects, key=squareness)
x, y, w, h = target
cx, cy = x + w / 2.0, y + h / 2.0
radius = math.hypot(w, h) / 2.0 + 12
print(f"target -> center ({cx:.1f},{cy:.1f}) radius {radius:.1f}")

# --- Step 3: animate a red circle sweeping around the winner --------------------
SS = 4  # supersampling for smooth anti-aliased stroke
LINE_W = 4
RED = (230, 30, 30)

def circle_layer(frac):
    """Return RGBA layer with the arc drawn from 0 to frac*360 degrees."""
    layer = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    if frac <= 0:
        return layer.resize((W, H), Image.LANCZOS)
    d = ImageDraw.Draw(layer)
    bbox = [(cx - radius) * SS, (cy - radius) * SS, (cx + radius) * SS, (cy + radius) * SS]
    start = -90
    end = start + 360 * min(frac, 1.0)
    if frac >= 1.0:
        d.ellipse(bbox, outline=RED + (255,), width=LINE_W * SS)
    else:
        d.arc(bbox, start=start, end=end, fill=RED + (255,), width=LINE_W * SS)
    return layer.resize((W, H), Image.LANCZOS)

HOLD_START = 8    # frames of "thinking" before drawing begins
SWEEP_END = 42    # circle finished by this frame, then hold

frames = []
for i in range(N_FRAMES):
    if i < HOLD_START:
        frac = 0.0
    elif i >= SWEEP_END:
        frac = 1.0
    else:
        t = (i - HOLD_START) / (SWEEP_END - HOLD_START)
        frac = 0.5 - 0.5 * math.cos(math.pi * t)  # ease in/out
    img = Image.fromarray(base.copy()).convert("RGBA")
    img.alpha_composite(circle_layer(frac))
    frames.append(np.array(img.convert("RGB")))

# --- Encode with ffmpeg (H.264, yuv420p) ----------------------------------------
cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
       "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "medium",
       "-movflags", "+faststart", OUT]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for f in frames:
    p.stdin.write(np.ascontiguousarray(f).tobytes())
p.stdin.close()
p.wait()
print("wrote", OUT, "frames:", len(frames))
