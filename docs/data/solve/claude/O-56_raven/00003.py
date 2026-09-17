"""Fill the bottom-right cell of the Raven matrix (two teal diamonds, as rows 1 and 3 match).

The '?' fades out while the answer diamonds grow/fade in. Every other pixel is untouched.
"""
import subprocess, shutil, os
import numpy as np
from PIL import Image

SRC = '/app/first_frame.png'
OUT_DIR = '/app/output'
OUT = os.path.join(OUT_DIR, 'video.mp4')
N_FRAMES, FPS = 35, 16

base = np.array(Image.open(SRC).convert('RGB')).astype(np.float32)
H, W, _ = base.shape

# Cell geometry (measured): grid lines at y 339-342, 680-683; x 340-341, 681-682.
# Bottom-right cell interior:
CY0, CY1, CX0, CX1 = 684, 1021, 683, 1022
# The answer = top-right cell content shifted down by two cell pitches (682 px).
ROW_SHIFT = 682
src_patch = base[CY0 - ROW_SHIFT:CY1 - ROW_SHIFT, CX0:CX1]          # top-right cell interior
cell_bg = base[CY0:CY1, CX0:CX1].copy()
cell_bg[:] = 255.0                                                  # blank white cell (no '?')
final_cell = src_patch.copy()

# Answer content bounding box inside the cell (for a grow-from-centre effect)
mask = np.abs(final_cell - 255).sum(2) > 30
ys, xs = np.where(mask)
ay0, ay1, ax0, ax1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1

def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1))

def answer_cell(s, alpha):
    """Answer content scaled by s about its own centre, blended with alpha over white."""
    if s >= 1.0 and alpha >= 1.0:
        return final_cell
    crop = Image.fromarray(final_cell[ay0:ay1, ax0:ax1].astype(np.uint8))
    w, h = crop.size
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    small = np.array(crop.resize((nw, nh), Image.LANCZOS)).astype(np.float32)
    out = np.full_like(final_cell, 255.0)
    cy, cx = (ay0 + ay1) / 2, (ax0 + ax1) / 2
    y0, x0 = int(round(cy - nh / 2)), int(round(cx - nw / 2))
    out[y0:y0 + nh, x0:x0 + nw] = small
    return 255.0 + (out - 255.0) * alpha

frames = []
for i in range(N_FRAMES):
    t = i / (N_FRAMES - 1)
    img = base.copy()
    if i == 0:
        frames.append(img.astype(np.uint8)); continue
    # '?' fades out over the first ~45% of the video
    q_alpha = 1.0 - ease(t / 0.45)
    cell = 255.0 + (base[CY0:CY1, CX0:CX1] - 255.0) * q_alpha
    # diamonds grow + fade in from 30% to 100%
    a = ease((t - 0.30) / 0.70)
    if a > 0:
        ans = answer_cell(0.4 + 0.6 * a, a) if i < N_FRAMES - 1 else final_cell
        cell = cell + (ans - 255.0)          # both are "ink over white": add darkness
    img[CY0:CY1, CX0:CX1] = np.clip(cell, 0, 255)
    frames.append(img.astype(np.uint8))

# Last frame is exactly base with the cell replaced
assert np.array_equal(frames[-1][CY0:CY1, CX0:CX1], final_cell.astype(np.uint8))
assert np.array_equal(frames[0], base.astype(np.uint8))

os.makedirs(OUT_DIR, exist_ok=True)
tmp = os.path.join(OUT_DIR, '_frames')
shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
for i, f in enumerate(frames):
    Image.fromarray(f).save(f'{tmp}/{i:04d}.png')
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS),
                '-i', f'{tmp}/%04d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-crf', '12', '-preset', 'slow', '-g', '1', OUT], check=True)
shutil.rmtree(tmp)
print('wrote', OUT)
