import numpy as np
from PIL import Image, ImageDraw
import subprocess, os, math

SRC = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
N = 60
FPS = 16

base = Image.open(SRC).convert('RGB')
arr = np.array(base).astype(int)
W, H = base.size

# --- detect circles: cluster non-background pixels by colour ---
bg = arr[0, 0]
mask = np.abs(arr - bg).sum(axis=2) > 40
ys, xs = np.nonzero(mask)
cols = arr[ys, xs]
# quantize colours to group
keys = (cols // 24)
uniq, inv = np.unique(keys, axis=0, return_inverse=True)
circles = []
for k in range(len(uniq)):
    sel = inv.ravel() == k
    if sel.sum() < 2000:
        continue
    cx, cy = xs[sel].mean(), ys[sel].mean()
    r = math.sqrt(sel.sum() / math.pi)
    circles.append((cx, cy, r))

# --- find the touching pair ---
best = None
for i in range(len(circles)):
    for j in range(i + 1, len(circles)):
        x1, y1, r1 = circles[i]; x2, y2, r2 = circles[j]
        d = math.hypot(x2 - x1, y2 - y1)
        gap = abs(d - (r1 + r2))
        if best is None or gap < best[0]:
            best = (gap, i, j)
_, i, j = best
x1, y1, r1 = circles[i]; x2, y2, r2 = circles[j]
d = math.hypot(x2 - x1, y2 - y1)
tx = x1 + (x2 - x1) * r1 / d
ty = y1 + (y2 - y1) * r1 / d
print('circles', circles, 'tangent', (tx, ty))

R = 22          # radius of the marker ring
LW = 5          # line width
SS = 4          # supersample for smooth strokes

def frame(t):
    """t in [0,1]: progress of drawing the black ring."""
    img = base.copy()
    if t <= 0:
        return img
    layer = Image.new('RGBA', (W * SS, H * SS), (0, 0, 0, 0))
    dr = ImageDraw.Draw(layer)
    bbox = [(tx - R) * SS, (ty - R) * SS, (tx + R) * SS, (ty + R) * SS]
    start = -90
    end = start + 360 * min(t, 1.0)
    if t >= 1.0:
        dr.ellipse(bbox, outline=(0, 0, 0, 255), width=LW * SS)
    else:
        dr.arc(bbox, start=start, end=end, fill=(0, 0, 0, 255), width=LW * SS)
    layer = layer.resize((W, H), Image.LANCZOS)
    img.paste(layer, (0, 0), layer)
    return img

os.makedirs('/app/output', exist_ok=True)
tmp = '/app/output/frames'
os.makedirs(tmp, exist_ok=True)
for f in os.listdir(tmp):
    os.remove(os.path.join(tmp, f))

hold_start = 8            # frames identical to the first frame
draw_frames = 40          # frames animating the ring
for k in range(N):
    if k < hold_start:
        t = 0.0
    else:
        t = min(1.0, (k - hold_start + 1) / draw_frames)
    frame(t).save(f'{tmp}/f{k:03d}.png')

subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS),
                '-i', f'{tmp}/f%03d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-crf', '15', OUT], check=True)
print('wrote', OUT)
