import numpy as np, subprocess, os
from PIL import Image

SRC = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
N_FRAMES, FPS = 60, 16
RED = np.array([255, 0, 0], np.uint8)

img = np.array(Image.open(SRC).convert('RGB'))
H, W, _ = img.shape
OUTLINES = [(90, 130, 70), (70, 130, 180), (60, 100, 140)]  # green, steel blue, dark blue

def fit_circle(pts):
    x, y = pts[:, 1].astype(float), pts[:, 0].astype(float)
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    b = x ** 2 + y ** 2
    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    return cx, cy, np.sqrt(c + cx ** 2 + cy ** 2)

circles = []
for col in OUTLINES:
    d = np.abs(img.astype(int) - np.array(col)).sum(2)
    circles.append(fit_circle(np.argwhere(d < 30)))

yy, xx = np.mgrid[0:H, 0:W]
inside = np.ones((H, W), bool)
for cx, cy, r in circles:
    inside &= (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
white = (img == 255).all(2)
region = inside & white  # keep outline / anti-aliased pixels untouched

# progressive fill: sweep from top of the region to the bottom
ys = np.where(region.any(1))[0]
y0, y1 = ys.min(), ys.max() + 1
start, end = 4, N_FRAMES - 6  # hold first/last frames

os.makedirs(os.path.dirname(OUT), exist_ok=True)
ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                       '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264',
                       '-pix_fmt', 'yuv420p', '-crf', '12', OUT], stdin=subprocess.PIPE)
for i in range(N_FRAMES):
    t = np.clip((i - start) / (end - start), 0, 1)
    t = t * t * (3 - 2 * t)  # ease in/out
    ylim = y0 + t * (y1 - y0)
    frame = img.copy()
    m = region & (yy < ylim)
    frame[m] = RED
    ff.stdin.write(frame.tobytes())
ff.stdin.close(); ff.wait()
print('circles:', [tuple(round(v, 1) for v in c) for c in circles], 'region px:', region.sum())
