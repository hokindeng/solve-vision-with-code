#!/usr/bin/env python3
"""Rearrange 7 circles into a centered horizontal row sorted by circumference (desc)."""
import os, subprocess
import numpy as np
import cv2
from PIL import Image

SRC = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
FPS, N_FRAMES = 16, 80
HOLD_START, HOLD_END = 4, 8      # frames held still at start / end
GAP = 30                          # horizontal gap between neighbouring circles

img = np.array(Image.open(SRC).convert('RGB'))
H, W = img.shape[:2]
bg_color = img[0, 0]

# --- detect circles as connected components of non-background pixels -------
diff = np.abs(img.astype(int) - bg_color.astype(int)).sum(2)
mask = (diff > 10).astype(np.uint8)
n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)

circles = []
for i in range(1, n):
    x, y, w, h, area = stats[i]
    if area < 200:
        continue
    comp = (lab[y:y + h, x:x + w] == i)
    sprite = img[y:y + h, x:x + w].copy()
    circles.append(dict(x=x, y=y, w=w, h=h, mask=comp, sprite=sprite,
                        r=np.sqrt(area / np.pi)))

# background = original with circle pixels removed
background = img.copy()
background[mask.astype(bool)] = bg_color

# --- target layout: sorted by circumference (i.e. radius), largest first ----
order = sorted(circles, key=lambda c: -c['r'])
total_w = sum(c['w'] for c in order) + GAP * (len(order) - 1)
x0 = (W - total_w) / 2.0
cy = H / 2.0
for c in order:
    c['tx'] = int(round(x0))                      # target top-left x
    c['ty'] = int(round(cy - c['h'] / 2.0))       # target top-left y
    x0 += c['w'] + GAP

def ease(t):  # smooth ease-in-out
    return t * t * (3 - 2 * t)

def compose(t):
    frame = background.copy()
    for c in order:
        px = int(round(c['x'] + (c['tx'] - c['x']) * t))
        py = int(round(c['y'] + (c['ty'] - c['y']) * t))
        h, w = c['h'], c['w']
        region = frame[py:py + h, px:px + w]
        region[c['mask']] = c['sprite'][c['mask']]
    return frame

os.makedirs(os.path.dirname(OUT), exist_ok=True)
move_frames = N_FRAMES - HOLD_START - HOLD_END
proc = subprocess.Popen(
    ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
     '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
     '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'medium',
     '-movflags', '+faststart', OUT], stdin=subprocess.PIPE)
for f in range(N_FRAMES):
    if f < HOLD_START:
        t = 0.0
    elif f >= N_FRAMES - HOLD_END:
        t = 1.0
    else:
        t = ease((f - HOLD_START) / (move_frames - 1))
    frame = compose(t) if t > 0 else img
    proc.stdin.write(np.ascontiguousarray(frame).tobytes())
proc.stdin.close()
proc.wait()
print('wrote', OUT)
