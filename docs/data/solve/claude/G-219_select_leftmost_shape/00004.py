"""Animate a red circle being drawn around the leftmost shape in first_frame.png."""
import os
import subprocess
import numpy as np
from PIL import Image
from scipy import ndimage

APP = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 48
SS = 4  # supersampling factor for anti-aliased stroke

base = np.array(Image.open(SRC).convert("RGB"))
H, W, _ = base.shape

# --- locate shapes and pick the leftmost one ------------------------------
bg = base[0, 0].astype(int)
mask = np.abs(base.astype(int) - bg).sum(2) > 30
lab, n = ndimage.label(mask)
shapes = []
for i in range(1, n + 1):
    ys, xs = np.nonzero(lab == i)
    if len(xs) < 50:
        continue
    shapes.append((xs.min(), xs.max(), ys.min(), ys.max()))
x0, x1, y0, y1 = min(shapes, key=lambda s: s[0])  # leftmost by left edge
cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
radius = 0.5 * np.hypot(x1 - x0, y1 - y0) + 14  # enclose shape with margin
thickness = 6
RED = np.array([255, 0, 0], dtype=np.float32)

# --- stroke mask for a partial arc, anti-aliased via supersampling --------
yy, xx = np.mgrid[0:H * SS, 0:W * SS]
px = (xx + 0.5) / SS - cx
py = (yy + 0.5) / SS - cy
dist = np.hypot(px, py)
ring = np.abs(dist - radius) <= thickness / 2.0
ang = (np.degrees(np.arctan2(py, px)) + 90.0) % 360.0  # start at top, clockwise


def arc_alpha(sweep_deg):
    """Return HxW alpha (0..1) of the arc covering [0, sweep_deg] degrees."""
    if sweep_deg <= 0:
        return np.zeros((H, W), np.float32)
    m = ring & (ang <= sweep_deg)
    if sweep_deg >= 360:
        m = ring
    else:  # round caps at both ends
        for a in (0.0, sweep_deg):
            t = np.radians(a - 90.0)
            ex, ey = cx + radius * np.cos(t), cy + radius * np.sin(t)
            m |= np.hypot((xx + 0.5) / SS - ex, (yy + 0.5) / SS - ey) <= thickness / 2.0
    return m.reshape(H, SS, W, SS).mean(axis=(1, 3)).astype(np.float32)


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)


os.makedirs(OUT_DIR, exist_ok=True)
frames = []
DRAW_FRAMES = 42  # finish the circle a bit before the end, then hold
for i in range(N_FRAMES):
    t = min(1.0, i / (DRAW_FRAMES - 1))
    a = arc_alpha(360.0 * ease(t))[..., None]
    frame = base.astype(np.float32) * (1 - a) + RED * a
    frames.append(np.clip(frame + 0.5, 0, 255).astype(np.uint8))

assert np.array_equal(frames[0], base), "first frame must match first_frame.png"

# --- encode with ffmpeg ----------------------------------------------------
cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
       "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
       "-movflags", "+faststart", OUT]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for f in frames:
    p.stdin.write(f.tobytes())
p.stdin.close()
p.wait()
if p.returncode != 0:
    raise SystemExit("ffmpeg failed")
print(f"wrote {OUT}: circle at ({cx:.0f},{cy:.0f}) r={radius:.0f}")
