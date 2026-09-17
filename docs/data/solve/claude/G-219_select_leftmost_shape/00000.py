import numpy as np, subprocess, os
from PIL import Image, ImageDraw
from scipy import ndimage

base = Image.open('/app/first_frame.png').convert('RGB')
arr = np.array(base)
# find leftmost shape (connected non-white component with smallest min-x)
mask = (arr < 240).any(axis=2)
lab, n = ndimage.label(mask)
best = None
for i in range(1, n + 1):
    ys, xs = np.where(lab == i)
    if len(xs) < 50:
        continue
    if best is None or xs.min() < best[0]:
        best = (xs.min(), xs, ys)
_, xs, ys = best
cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
half = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2).max()  # farthest pixel from center
r = half + 20  # margin around the shape
W = 6          # line width
N = 48
os.makedirs('/app/output', exist_ok=True)
frames = []
for k in range(N):
    im = base.copy()
    if k > 0:
        # draw arc progressively; full circle at last frame
        t = k / (N - 1)
        end = -90 + 360 * t
        d = ImageDraw.Draw(im)
        box = [cx - r, cy - r, cx + r, cy + r]
        if t >= 1:
            d.ellipse(box, outline=(255, 0, 0), width=W)
        else:
            d.arc(box, start=-90, end=end, fill=(255, 0, 0), width=W)
    frames.append(np.array(im))

tmp = '/app/output/frames'
os.makedirs(tmp, exist_ok=True)
for i, f in enumerate(frames):
    Image.fromarray(f).save(f'{tmp}/{i:03d}.png')
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '16', '-i', f'{tmp}/%03d.png',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '15', '/app/output/video.mp4'], check=True)
for fn in os.listdir(tmp):
    os.remove(f'{tmp}/{fn}')
os.rmdir(tmp)
print('done', cx, cy, r)
