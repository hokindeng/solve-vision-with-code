"""Generate video: the green-bordered decagon slides horizontally to sit
directly below the teal square with the red star. Everything else is unchanged."""
import os, subprocess, numpy as np
from PIL import Image

SRC = "/app/first_frame.png"
OUT_DIR = "/app/output"
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 60, 16

base = np.array(Image.open(SRC).convert("RGB"))
H, W, _ = base.shape
nonwhite = base.sum(-1) != 765

# --- locate the moving object (green border) ---
g = (base[..., 1] > 180) & (base[..., 0] < 60) & (base[..., 2] < 60)
ys, xs = np.where(g)
y0, y1 = ys.min(), ys.max()
# everything non-white in that row band belongs to the object (verified: band is otherwise empty)
band = nonwhite[y0:y1 + 1]
bx = np.where(band.any(0))[0]
x0, x1 = bx.min(), bx.max()
obj_mask = band[:, x0:x1 + 1]
obj_rgb = base[y0:y1 + 1, x0:x1 + 1].copy()
obj_cx = (x0 + x1) / 2.0

# --- locate the target (red star center) ---
r = (base[..., 0] > 230) & (base[..., 1] < 30) & (base[..., 2] < 30)
rys, rxs = np.where(r)
star_cx = (rxs.min() + rxs.max()) / 2.0
dx_total = int(round(star_cx - obj_cx))

# background with the object removed (its footprint is pure white)
bg = base.copy()
bg[y0:y1 + 1, x0:x1 + 1][obj_mask] = 255

def ease(t):  # smoothstep ease-in-out
    return t * t * (3 - 2 * t)

os.makedirs(OUT_DIR, exist_ok=True)
frames = []
for i in range(N_FRAMES):
    t = i / (N_FRAMES - 1)
    dx = int(round(dx_total * ease(t)))
    f = bg.copy()
    region = f[y0:y1 + 1, x0 + dx:x1 + 1 + dx]
    region[obj_mask] = obj_rgb[obj_mask]
    frames.append(f)

# first frame must equal source exactly
assert np.array_equal(frames[0], base)

raw = np.concatenate(frames).tobytes()
subprocess.run([
    "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
    "-s", f"{W}x{H}", "-r", str(FPS), "-i", "pipe:",
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT
], input=raw, check=True)
Image.fromarray(frames[-1]).save(os.path.join(OUT_DIR, "last_frame.png"))
print(f"wrote {OUT}: {N_FRAMES} frames, dx={dx_total}")
