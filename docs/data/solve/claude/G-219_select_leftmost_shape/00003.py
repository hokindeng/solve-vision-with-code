import os, subprocess, math
import numpy as np
from PIL import Image, ImageDraw

BASE = "/app/first_frame.png"
OUT_DIR = "/app/output"
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES, SS = 16, 48, 4  # SS = supersampling factor for anti-aliasing

base = np.array(Image.open(BASE).convert("RGB"))
H, W = base.shape[:2]

# --- locate shapes and pick the leftmost one -------------------------------
bg = base[0, 0].astype(int)
mask = np.abs(base.astype(int) - bg).sum(axis=2) > 30
from scipy import ndimage
lab, n = ndimage.label(mask)
shapes = []
for i in range(1, n + 1):
    ys, xs = np.where(lab == i)
    if len(xs) < 200:
        continue
    shapes.append(dict(x0=xs.min(), x1=xs.max(), y0=ys.min(), y1=ys.max()))
left = min(shapes, key=lambda s: s["x0"])
cx = (left["x0"] + left["x1"]) / 2.0
cy = (left["y0"] + left["y1"]) / 2.0
half = max(left["x1"] - left["x0"], left["y1"] - left["y0"]) / 2.0
radius = half * math.sqrt(2) * 1.02 + 4   # encloses the bounding box with a small margin
stroke = 6
RED = np.array([230, 30, 30], dtype=np.float32)

def frame(progress):
    """progress in [0,1]: fraction of the circle drawn (clockwise from top)."""
    img = base.copy()
    if progress <= 0:
        return img
    layer = Image.new("L", (W * SS, H * SS), 0)
    d = ImageDraw.Draw(layer)
    bbox = [(cx - radius) * SS, (cy - radius) * SS, (cx + radius) * SS, (cy + radius) * SS]
    start = -90
    end = start + 360 * progress
    if progress >= 1:
        d.ellipse(bbox, outline=255, width=stroke * SS)
    else:
        d.arc(bbox, start=start, end=end, fill=255, width=stroke * SS)
    alpha = np.array(layer.resize((W, H), Image.LANCZOS)).astype(np.float32) / 255.0
    alpha = alpha[..., None]
    out = img.astype(np.float32) * (1 - alpha) + RED * alpha
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)

os.makedirs(OUT_DIR, exist_ok=True)
ff = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
     "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "slow", OUT],
    stdin=subprocess.PIPE)
for k in range(N_FRAMES):
    t = k / (N_FRAMES - 1)
    p = t * t * (3 - 2 * t)  # smoothstep easing: starts at 0, ends at 1
    ff.stdin.write(frame(p).tobytes())
ff.stdin.close()
ff.wait()
print("wrote", OUT, "center", (cx, cy), "radius", radius)
