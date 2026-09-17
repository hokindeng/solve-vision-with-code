#!/usr/bin/env python3
"""Complete the Raven-style matrix: each row is a permutation of
{square, triangle, diamond}; row 3 has diamond, square -> missing triangle.
The triangle is copied pixel-exactly from the row-2/col-1 triangle and traced
progressively into the bottom-right cell while the '?' fades out."""
import subprocess, os
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 35, 16

base = np.array(Image.open(SRC).convert("RGB"))
H, W, _ = base.shape

# Cell geometry (grid lines at 0, 340-341, 681-682, 1022-1023)
# Source triangle: row 1, col 0.  Target: row 2, col 2.
src = base[342:681, 0:340]
tri_mask_src = (src == [0, 150, 0]).all(2)
ys, xs = np.nonzero(tri_mask_src)
dx, dy = 682, 342 + 341            # xs are absolute; ys are relative to crop at y=342
tx, ty = xs + dx, ys + dy
tri_color = np.array([0, 150, 0], np.uint8)

# Question-mark mask inside target cell (any non-white pixel in the cell interior)
cell = base[684:1021, 684:1021]
q_mask_local = (cell != 255).any(2)
qys, qxs = np.nonzero(q_mask_local)
qy, qx = qys + 684, qxs + 684
q_colors = base[qy, qx].astype(float)

# Order triangle pixels along the perimeter (apex -> bottom-right -> bottom-left -> apex)
x0, x1, y0, y1 = tx.min(), tx.max(), ty.min(), ty.max()
apex = np.array([(x0 + x1) / 2.0, y0]); br = np.array([x1, y1]); bl = np.array([x0, y1])
verts = [apex, br, bl, apex]
seg_len = [np.linalg.norm(verts[i + 1] - verts[i]) for i in range(3)]
total = sum(seg_len)
pts = np.stack([tx, ty], 1).astype(float)
best_t = np.full(len(pts), np.inf); best_d = np.full(len(pts), np.inf); acc = 0.0
for i in range(3):
    a, b = verts[i], verts[i + 1]; ab = b - a; L = seg_len[i]
    u = np.clip(((pts - a) @ ab) / (L * L), 0, 1)
    proj = a + u[:, None] * ab
    d = np.linalg.norm(pts - proj, axis=1)
    better = d < best_d
    best_d[better] = d[better]; best_t[better] = (acc + u[better] * L) / total
    acc += L

def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1))

os.makedirs(OUT_DIR, exist_ok=True)
frames = []
for f in range(N_FRAMES):
    img = base.copy()
    p = f / (N_FRAMES - 1)
    # '?' fades to white during the first ~40% of the clip
    fade = ease(p / 0.4)
    if fade > 0:
        img[qy, qx] = np.clip(q_colors * (1 - fade) + 255 * fade, 0, 255).astype(np.uint8)
    # triangle traced from 10% to 100%
    draw = ease((p - 0.1) / 0.9)
    if f == N_FRAMES - 1:
        draw = 1.0
    sel = best_t <= draw + 1e-9 if draw > 0 else np.zeros(len(tx), bool)
    img[ty[sel], tx[sel]] = tri_color
    frames.append(img)

# Sanity: last frame has full triangle and no '?', first frame is untouched
assert (frames[0] == base).all()
m_out = np.ones((H, W), bool); m_out[684:1021, 684:1021] = False
assert all((fr[m_out] == base[m_out]).all() for fr in frames)
assert ty.min() == 794 and ty.max() == 909 and tx.min() == 794 and tx.max() == 909
assert (frames[-1][ty, tx] == tri_color).all()

ff = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
     "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-qp", "0", "-preset", "veryslow", OUT],
    stdin=subprocess.PIPE)
for fr in frames:
    ff.stdin.write(fr.tobytes())
ff.stdin.close(); ff.wait()
print("wrote", OUT, "frames:", len(frames))
