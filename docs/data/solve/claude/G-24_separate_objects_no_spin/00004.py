#!/usr/bin/env python3
"""Move the 4 left-side objects horizontally into their dashed target outlines."""
import subprocess, numpy as np
from PIL import Image
from scipy import ndimage

SRC = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
N_FRAMES, FPS = 30, 16
SPLIT_X = 520  # objects live left of this, targets right of it

img = np.array(Image.open(SRC).convert('RGB'))
H, W, _ = img.shape
nonwhite = np.any(img != 255, axis=2)
xs = np.arange(W)[None, :]

# --- objects: connected non-white components on the left ---
lab, n = ndimage.label(nonwhite & (xs < SPLIT_X))
objects = []
for i in range(1, n + 1):
    m = lab == i
    if m.sum() < 200:
        continue
    objects.append(m)

# dashed target pixels on the right
target_px = nonwhite & (xs >= SPLIT_X)

# --- background: frame with objects erased (scene background is white) ---
bg = img.copy()
for m in objects:
    bg[m] = 255

# --- find horizontal shift per object by matching its outline to the dashes ---
def outline(mask):
    return mask & ~ndimage.binary_erosion(mask, iterations=2)

def best_shift(mask):
    ol = outline(mask)
    band = ndimage.binary_dilation(target_px, iterations=2)
    ys, xs_ = np.nonzero(ol)
    best, best_dx = -1, 0
    for dx in range(0, W - xs_.max()):
        score = band[ys, xs_ + dx].sum()
        if score > best:
            best, best_dx = score, dx
    return best_dx

shifts = [best_shift(m) for m in objects]
for m, dx in zip(objects, shifts):
    ys, xs_ = np.nonzero(m)
    print(f'object bbox x[{xs_.min()},{xs_.max()}] y[{ys.min()},{ys.max()}] -> dx={dx}')

def ease(t):  # smooth ease-in-out
    return t * t * (3 - 2 * t)

def render(t):
    frame = bg.copy()
    for m, dx in zip(objects, shifts):
        d = int(round(dx * ease(t)))
        ys, xs_ = np.nonzero(m)
        frame[ys, xs_ + d] = img[ys, xs_]
    return frame

frames = [render(i / (N_FRAMES - 1)) for i in range(N_FRAMES)]
assert np.array_equal(frames[0], img), 'first frame must match first_frame.png'

proc = subprocess.Popen(
    ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
     '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
     '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '12', '-preset', 'slow', OUT],
    stdin=subprocess.PIPE)
for f in frames:
    proc.stdin.write(np.ascontiguousarray(f).tobytes())
proc.stdin.close(); proc.wait()
Image.fromarray(frames[-1]).save('/app/output/last_frame.png')
print('wrote', OUT)
