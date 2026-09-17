import numpy as np, cv2, subprocess, os
from PIL import Image

SRC = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
N_FRAMES, FPS = 37, 16

img = np.array(Image.open(SRC).convert('RGB'))
H, W = img.shape[:2]

def fit_circle(mask):
    """Least-squares circle fit to the outer contour of a mask (ignores dents from dots)."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)
    x, y = pts[:, 0], pts[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    b = x**2 + y**2
    for _ in range(3):  # robust refit: drop outliers (dents where dots sit on the edge)
        cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
        r = np.sqrt(c + cx**2 + cy**2)
        res = np.abs(np.hypot(x - cx, y - cy) - r)
        keep = res < max(1.5, 2 * np.median(res))
        A, b, x, y = A[keep], b[keep], x[keep], y[keep]
    return cx, cy, r

def color_mask(rgb, tol=40):
    return (np.abs(img.astype(int) - np.array(rgb)).sum(2) < tol)

purple = color_mask((212, 170, 251)); green = color_mask((128, 212, 128)); ovl = color_mask((148, 148, 188))
black = color_mask((0, 0, 0), 150)
# fill dots so shape masks are solid
k = np.ones((3, 3), np.uint8)
def solid(m):
    m = (m | black).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    return m
c1 = fit_circle(solid(purple | ovl))
c2 = fit_circle(solid(green | ovl))

# detect dots
n, lab, stats, cents = cv2.connectedComponentsWithStats(black.astype(np.uint8), 8)
dots = []
for i in range(1, n):
    a = stats[i, cv2.CC_STAT_AREA]
    if 10 < a < 400:
        dots.append((cents[i][0], cents[i][1], np.sqrt(a / np.pi)))

def inside(d, c, margin=1.0):
    return np.hypot(d[0] - c[0], d[1] - c[1]) + d[2] + margin < c[2]
targets = [d for d in dots if inside(d, c1) and inside(d, c2)]
targets.sort(key=lambda d: (d[1], d[0]))
print('circles:', c1, c2, 'dots:', len(dots), 'targets:', len(targets))

# animation: hold, then each red circle grows in sequentially, finishing before the end
RED = (230, 30, 30)
R_FINAL = 18
hold, tail = 3, 3
avail = N_FRAMES - hold - tail
per = avail / max(1, len(targets))
grow = max(2, int(per * 0.8))

def frame(t):
    f = img.copy()
    for j, d in enumerate(targets):
        start = hold + j * per
        p = np.clip((t - start) / grow, 0, 1)
        if p <= 0:
            continue
        p = 1 - (1 - p) ** 2
        r = 4 + (R_FINAL - 4) * p
        cv2.circle(f, (int(round(d[0] * 16)), int(round(d[1] * 16))), int(round(r * 16)), RED, 3, cv2.LINE_AA, shift=4)
    return f

os.makedirs(os.path.dirname(OUT), exist_ok=True)
tmp = '/app/output/_frames'; os.makedirs(tmp, exist_ok=True)
for t in range(N_FRAMES):
    Image.fromarray(frame(t)).save(f'{tmp}/{t:03d}.png')
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS), '-i', f'{tmp}/%03d.png',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '12', OUT], check=True)
for fn in os.listdir(tmp): os.remove(f'{tmp}/{fn}')
os.rmdir(tmp)
print('wrote', OUT)
