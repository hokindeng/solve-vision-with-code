#!/usr/bin/env python3
"""Move the green-bordered hexagon horizontally so it sits directly below the red star."""
import subprocess, numpy as np
from PIL import Image

FRAMES, FPS = 60, 16
src = np.array(Image.open('/app/first_frame.png').convert('RGB'))
H, W, _ = src.shape
r, g, b = (src[..., i].astype(int) for i in range(3))

# --- locate the green border and the red star -------------------------------
green = (g > 180) & (r < 60) & (b < 60)
ys, xs = np.nonzero(green)
x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
obj_cx = (x0 + x1) / 2.0

red = (r > 200) & (g < 60) & (b < 60)
sy, sx = np.nonzero(red)
star_cx = (sx.min() + sx.max()) / 2.0

dx_total = int(round(star_cx - obj_cx))

# --- object mask: convex fill of every non-white pixel inside the border bbox --
nonwhite = np.any(src != 255, axis=2)
mask = np.zeros((H, W), bool)
for y in range(y0, y1 + 1):
    row = np.nonzero(nonwhite[y, x0:x1 + 1])[0]
    if row.size:
        mask[y, x0 + row.min():x0 + row.max() + 1] = True

background = src.copy()
background[mask] = 255                      # white canvas where the object was
obj_pixels = src[mask]
my, mx = np.nonzero(mask)

def ease(t):                                # smooth ease-in-out
    return 0.5 - 0.5 * np.cos(np.pi * t)

# --- render frames and pipe to ffmpeg ----------------------------------------
cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
       '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
       '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '12', '/app/output/video.mp4']
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for i in range(FRAMES):
    t = i / (FRAMES - 1)
    dx = int(round(dx_total * ease(t)))
    frame = background.copy()
    frame[my, mx + dx] = obj_pixels          # object + green border move together
    proc.stdin.write(frame.tobytes())
proc.stdin.close()
proc.wait()
print(f'object center x {obj_cx} -> star x {star_cx}, dx={dx_total}; wrote /app/output/video.mp4')
