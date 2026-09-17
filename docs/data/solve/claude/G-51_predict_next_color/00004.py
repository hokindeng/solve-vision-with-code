"""Predict the next color in the sequence (teal, teal, purple, purple, ?) -> teal.
Animates the empty outline triangle filling with teal, bottom to top, over 64 frames."""
import numpy as np
from PIL import Image
from collections import deque
import subprocess, os, shutil

SRC = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
TMP = '/app/output/frames'
N_FRAMES, FPS = 64, 16
FILL = np.array([6, 99, 131], dtype=np.uint8)   # teal, same as triangles 1 & 2

base = np.array(Image.open(SRC).convert('RGB'))
H, W, _ = base.shape

# Interior of the outline triangle: flood-fill white pixels from its centroid.
seed = (540, 912)
white = (base == 255).all(axis=2)
interior = np.zeros((H, W), dtype=bool)
q = deque([seed]); interior[seed] = True
while q:
    y, x = q.popleft()
    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
        ny, nx = y+dy, x+dx
        if 0 <= ny < H and 0 <= nx < W and white[ny, nx] and not interior[ny, nx]:
            interior[ny, nx] = True; q.append((ny, nx))

ys = np.nonzero(interior.any(axis=1))[0]
y_top, y_bot = ys.min(), ys.max()

def ease(t):  # smoothstep
    return t*t*(3-2*t)

shutil.rmtree(TMP, ignore_errors=True); os.makedirs(TMP)
for i in range(N_FRAMES):
    t = ease(i/(N_FRAMES-1))
    frame = base.copy()
    if t > 0:
        level = y_bot - t*(y_bot - y_top + 1)           # fill rises from the bottom
        rows = np.arange(H)[:, None] >= level
        m = interior & rows
        frame[m] = FILL
    Image.fromarray(frame).save(f'{TMP}/{i:03d}.png')

subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS),
                '-i', f'{TMP}/%03d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-crf', '12', '-r', str(FPS), OUT], check=True)
shutil.rmtree(TMP)
print('wrote', OUT)
