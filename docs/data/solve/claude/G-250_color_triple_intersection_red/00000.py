import numpy as np, subprocess, os
from PIL import Image

src = Image.open('/app/first_frame.png').convert('RGB')
base = np.array(src).astype(np.uint8)
H, W = base.shape[:2]
N_FRAMES, FPS = 60, 16

# --- detect the three circle outlines by their distinct colors and fit circles ---
flat = base.reshape(-1, 3)
cols, counts = np.unique(flat, axis=0, return_counts=True)
ring_cols = [c for c, n in zip(cols, counts) if n > 500 and not (c == 255).all()]

def fit_circle(pts):
    x, y = pts[:, 0].astype(float), pts[:, 1].astype(float)
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    b = x**2 + y**2
    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    return cx, cy, np.sqrt(c + cx**2 + cy**2)

circles = []
for c in ring_cols:
    ys, xs = np.where((np.abs(base.astype(int) - c.astype(int)).sum(2) < 30))
    circles.append(fit_circle(np.c_[xs, ys]))
assert len(circles) == 3, circles

# --- triple intersection mask (only white pixels, leave ring strokes intact) ---
yy, xx = np.mgrid[0:H, 0:W]
inside = np.ones((H, W), bool)
for cx, cy, r in circles:
    inside &= (xx - cx)**2 + (yy - cy)**2 <= r**2
white = (base == 255).all(2)
region = inside & white

# --- animate: radial fill growing from the region centroid ---
ry, rx = np.where(region)
cy0, cx0 = ry.mean(), rx.mean()
dist = np.sqrt((xx - cx0)**2 + (yy - cy0)**2)
dmax = dist[region].max() + 1e-6
RED = np.array([255, 0, 0], np.uint8)

os.makedirs('/app/output', exist_ok=True)
frames = []
for i in range(N_FRAMES):
    t = 0.0 if i == 0 else min(1.0, (i / (N_FRAMES - 4)))  # first frame untouched, hold at end
    t = t * t * (3 - 2 * t)  # ease
    f = base.copy()
    m = region & (dist <= t * dmax * 1.001) if i > 0 else np.zeros_like(region)
    if i >= N_FRAMES - 4:
        m = region
    f[m] = RED
    frames.append(f)

p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                      '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                      '-crf', '12', '/app/output/video.mp4'], stdin=subprocess.PIPE)
for f in frames:
    p.stdin.write(f.tobytes())
p.stdin.close(); p.wait()
Image.fromarray(frames[-1]).save('/app/output/last_frame.png')
print('circles', circles, 'region px', region.sum())
