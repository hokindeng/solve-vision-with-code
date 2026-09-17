import numpy as np, os, subprocess
from PIL import Image, ImageDraw
from scipy import ndimage

W = H = 1024; FPS = 16; N = 48
base = Image.open('/app/first_frame.png').convert('RGB')
arr = np.array(base)

# find leftmost shape (connected non-white component with smallest min-x)
mask = (arr != 255).any(axis=2)
lab, n = ndimage.label(mask)
best = None
for i in range(1, n + 1):
    ys, xs = np.where(lab == i)
    if len(xs) < 50: continue
    if best is None or xs.min() < best[0]:
        best = (xs.min(), xs.max(), ys.min(), ys.max())
x0, x1, y0, y1 = best
cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
r = np.hypot(x1 - x0, y1 - y0) / 2 + 14   # circumscribing radius plus margin
r = min(r, cx - 3, cy - 3)                # stay inside canvas
thick = 6

os.makedirs('/app/output', exist_ok=True)
frames_dir = '/app/output/frames'; os.makedirs(frames_dir, exist_ok=True)
bbox = [cx - r, cy - r, cx + r, cy + r]
for f in range(N):
    im = base.copy()
    d = ImageDraw.Draw(im)
    if f > 0:
        t = f / (N - 1)
        sweep = 360 * t
        d.arc(bbox, start=-90, end=-90 + sweep, fill=(255, 0, 0), width=thick)
        # round the arc ends slightly so the stroke looks like a pen
        for ang in (-90, -90 + sweep):
            a = np.deg2rad(ang)
            px, py = cx + (r - thick / 2) * np.cos(a), cy + (r - thick / 2) * np.sin(a)
            d.ellipse([px - thick / 2, py - thick / 2, px + thick / 2, py + thick / 2], fill=(255, 0, 0))
    im.save(f'{frames_dir}/{f:03d}.png')

subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS),
                '-i', f'{frames_dir}/%03d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-crf', '15', '/app/output/video.mp4'], check=True)
print('done', cx, cy, r)
