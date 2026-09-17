import numpy as np
from PIL import Image
from scipy import ndimage
import subprocess, os, shutil, tempfile

SRC = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
W = H = 1024
FPS = 16
N = 40

img = np.array(Image.open(SRC).convert('RGB'))
bg = img[0, 0].copy()
mask = np.any(img != bg, axis=2)

# baseline: long thin horizontal component
lab, n = ndimage.label(ndimage.binary_dilation(mask, iterations=3))
objs = ndimage.find_objects(lab)
boxes = []
baseline_top = None
for i, sl in enumerate(objs):
    ys, xs = sl
    if xs.stop - xs.start > W * 0.6:
        rows = np.where(mask[ys, :].any(1))[0] + ys.start
        baseline_top = rows.min()
    else:
        boxes.append([ys.start, ys.stop, xs.start, xs.stop])

# merge boxes contained in other boxes (e.g. frog pupils inside eye whites)
def contains(a, b):
    return a[0] <= b[0] and a[1] >= b[1] and a[2] <= b[2] and a[3] >= b[3]
boxes.sort(key=lambda b: -(b[1]-b[0])*(b[3]-b[2]))
merged = []
for b in boxes:
    if not any(contains(m, b) for m in merged):
        merged.append(b)

sprites = []
for y0, y1, x0, x1 in merged:
    sub = mask[y0:y1, x0:x1]
    ys, xs = np.where(sub)
    ty0, ty1 = y0 + ys.min(), y0 + ys.max() + 1
    tx0, tx1 = x0 + xs.min(), x0 + xs.max() + 1
    rgb = img[ty0:ty1, tx0:tx1]
    a = mask[ty0:ty1, tx0:tx1]
    sprites.append(dict(x=tx0, y=ty0, w=tx1-tx0, h=ty1-ty0, rgb=rgb, a=a,
                        size=(tx1-tx0)*(ty1-ty0)))

# sort by size: smallest -> largest, left -> right
sprites.sort(key=lambda s: s['size'])
k = len(sprites)
centers = [W * (i + 1) / (k + 1) for i in range(k)]
bottom = baseline_top - 1  # face bottom rests just above the baseline
for s, cx in zip(sprites, centers):
    s['tx'] = int(round(cx - s['w'] / 2))
    s['ty'] = bottom - s['h']

# clean background: original frame with faces removed
clean = img.copy()
for s in sprites:
    clean[s['y']:s['y']+s['h'], s['x']:s['x']+s['w']][s['a']] = bg

def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)

def frame(i):
    hold0, hold1 = 3, 4  # brief hold at start and end
    t = np.clip((i - hold0) / (N - 1 - hold0 - hold1), 0, 1)
    e = ease(t)
    f = clean.copy()
    for s in sorted(sprites, key=lambda s: s['size']):
        x = int(round(s['x'] + (s['tx'] - s['x']) * e))
        y = int(round(s['y'] + (s['ty'] - s['y']) * e))
        region = f[y:y+s['h'], x:x+s['w']]
        region[s['a']] = s['rgb'][s['a']]
    return f

tmp = tempfile.mkdtemp()
for i in range(N):
    Image.fromarray(frame(i)).save(os.path.join(tmp, f'{i:04d}.png'))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS),
                '-i', os.path.join(tmp, '%04d.png'), '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p', '-crf', '12', OUT], check=True)
shutil.rmtree(tmp)
print('wrote', OUT, 'baseline_top', baseline_top, [(s['x'], s['y'], s['tx'], s['ty'], s['size']) for s in sprites])
