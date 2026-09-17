#!/usr/bin/env python3
"""Symbol deletion: fade out the red-bordered symbol, then slide the
following symbols left to close the gap. Everything else stays as in
first_frame.png."""
import subprocess
import numpy as np
from PIL import Image

W = H = 1024
FPS = 16
N_FRAMES = 46

base = np.array(Image.open('/app/first_frame.png').convert('RGB')).astype(np.float32)

# --- detect layout -------------------------------------------------------
gray = (np.abs(base - 160).sum(2) < 30)
red = (base[:, :, 0] > 200) & (base[:, :, 1] < 60) & (base[:, :, 2] < 60)
ry, rx = np.where(red)
red_box = (rx.min(), ry.min(), rx.max() + 1, ry.max() + 1)   # x0,y0,x1,y1

# cell borders: gray columns in the middle row of the cells
cy = (red_box[1] + red_box[3]) // 2
cols = np.where(gray[cy])[0]
# each cell contributes two gray columns (left/right border); pair them up
xs = []
prev = None
for c in cols:
    if prev is None or c - prev > 2:
        xs.append(c)
    prev = c
cells = [(xs[i], xs[i + 1] + 1) for i in range(0, len(xs), 2)]   # (x0, x1)
rows = np.where(gray[:, cells[0][0]])[0]
cy0, cy1 = rows.min(), rows.max() + 1

# the target cell's gray frame is hidden under the red border: it sits centred in it
cw = cells[0][1] - cells[0][0]
tx0 = (red_box[0] + red_box[2]) // 2 - cw // 2
cells.append((tx0, tx0 + cw))
cells.sort()
target = cells.index((tx0, tx0 + cw))
pitch = cells[1][0] - cells[0][0]

# --- sprites --------------------------------------------------------------
def crop(x0, y0, x1, y1):
    return base[y0:y1, x0:x1].copy()

cell_sprites = [crop(x0, cy0, x1, cy1) for (x0, x1) in cells]
red_sprite = crop(*red_box)

# background = frame with the red border region and all cells blanked to white
bg = base.copy()
bg[red_box[1]:red_box[3], red_box[0]:red_box[2]] = 255
for (x0, x1) in cells:
    bg[cy0:cy1, x0:x1] = 255

def paste(canvas, sprite, x, y, alpha=1.0):
    h, w = sprite.shape[:2]
    region = canvas[y:y + h, x:x + w]
    canvas[y:y + h, x:x + w] = region * (1 - alpha) + sprite * alpha

def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1))

# --- timeline -------------------------------------------------------------
FADE_END = 18      # frames 0..17: target (and red border) fade out
SLIDE_START = 20
SLIDE_END = 42     # frames 20..41: cells after target slide left

frames = []
for f in range(N_FRAMES):
    canvas = bg.copy()
    fade = 1.0 - ease(f / (FADE_END - 1)) if f < FADE_END else 0.0
    slide = ease((f - SLIDE_START) / (SLIDE_END - 1 - SLIDE_START)) if f >= SLIDE_START else 0.0

    for i, (x0, x1) in enumerate(cells):
        if i == target:
            if fade > 0:
                # shrink + fade the target symbol
                spr = cell_sprites[i]
                h, w = spr.shape[:2]
                s = 0.6 + 0.4 * fade
                nh, nw = max(2, int(round(h * s))), max(2, int(round(w * s)))
                small = np.array(Image.fromarray(spr.astype(np.uint8)).resize((nw, nh), Image.LANCZOS)).astype(np.float32)
                paste(canvas, small, x0 + (w - nw) // 2, cy0 + (h - nh) // 2, fade)
        elif i > target:
            x = int(round(x0 - pitch * slide))
            paste(canvas, cell_sprites[i], x, cy0)
        else:
            paste(canvas, cell_sprites[i], x0, cy0)

    if fade > 0:
        paste(canvas, red_sprite, red_box[0], red_box[1], fade)

    if f == 0:
        canvas = base.copy()   # exact first frame
    frames.append(np.clip(canvas, 0, 255).astype(np.uint8))

# --- encode ---------------------------------------------------------------
import os
os.makedirs('/app/output', exist_ok=True)
cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
       '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
       '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '/app/output/video.mp4']
p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for fr in frames:
    p.stdin.write(fr.tobytes())
p.stdin.close()
p.wait()
print('cells', cells, 'target', target, 'wrote', len(frames), 'frames')
