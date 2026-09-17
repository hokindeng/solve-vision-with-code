import numpy as np, subprocess, math
from PIL import Image, ImageDraw

SRC = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
N, FPS = 60, 16

base = Image.open(SRC).convert('RGB')
arr = np.array(base).astype(int)
W, H = base.size

# Detect filled circles: each distinct non-background colour is one disc.
bg = arr[0, 0]
cols, counts = np.unique(arr.reshape(-1, 3), axis=0, return_counts=True)
circles = []
for c, n in zip(cols, counts):
    if (c == bg).all() or n < 500:
        continue
    m = (np.abs(arr - c).sum(axis=2) < 30)
    ys, xs = np.nonzero(m)
    cx, cy = xs.mean(), ys.mean()
    r = math.sqrt(m.sum() / math.pi)
    circles.append((cx, cy, r))

# Pair whose centre distance best matches the sum of radii is tangent.
best = None
for i in range(len(circles)):
    for j in range(i + 1, len(circles)):
        x1, y1, r1 = circles[i]; x2, y2, r2 = circles[j]
        d = math.hypot(x2 - x1, y2 - y1)
        err = abs(d - (r1 + r2))
        if best is None or err < best[0]:
            best = (err, i, j, d)
_, i, j, d = best
x1, y1, r1 = circles[i]; x2, y2, r2 = circles[j]
tx = x1 + (x2 - x1) * r1 / d
ty = y1 + (y2 - y1) * r1 / d

R, LW = 40, 6   # radius and line width of the marker ring
S = 4           # supersampling for smooth anti-aliased ring

def frame(k):
    t = k / (N - 1)
    if t <= 0:
        return base.copy()
    sweep = 360 * min(1.0, t / 0.9)   # ring completes at 90% of duration, then holds
    ov = Image.new('L', (W * S, H * S), 0)
    dr = ImageDraw.Draw(ov)
    bbox = [(tx - R) * S, (ty - R) * S, (tx + R) * S, (ty + R) * S]
    if sweep >= 360:
        dr.ellipse(bbox, outline=255, width=LW * S)
    else:
        dr.arc(bbox, start=-90, end=-90 + sweep, fill=255, width=LW * S)
    mask = ov.resize((W, H), Image.LANCZOS)
    out = base.copy()
    out.paste(Image.new('RGB', (W, H), (0, 0, 0)), (0, 0), mask)
    return out

ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                       '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264',
                       '-pix_fmt', 'yuv420p', '-crf', '15', OUT], stdin=subprocess.PIPE)
for k in range(N):
    ff.stdin.write(np.array(frame(k)).tobytes())
ff.stdin.close(); ff.wait()
print('circles', [(round(a), round(b), round(c)) for a, b, c in circles], 'tangent', (round(tx, 1), round(ty, 1)))
