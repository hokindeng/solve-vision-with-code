import numpy as np, subprocess, os, tempfile
from PIL import Image
from scipy import ndimage

W = H = 1024; N_FRAMES = 35; FPS = 16
BG = np.array([220, 220, 220], np.uint8)

src = np.array(Image.open('/app/first_frame.png').convert('RGB'))
mask = np.any(src != BG, axis=2)
lab, n = ndimage.label(mask)
sizes = ndimage.sum(mask, lab, range(1, n + 1))
comps = [(i + 1, s) for i, s in enumerate(sizes)]

# Solid objects: the two biggest components. Outlines: dash fragments grouped by color.
solids = sorted(comps, key=lambda c: -c[1])[:2]

def bbox(m):
    ys, xs = np.where(m)
    return xs.min(), xs.max(), ys.min(), ys.max()

def dominant_color(m):
    cols, cnt = np.unique(src[m], axis=0, return_counts=True)
    return cols[cnt.argmax()]

def ref_center(x0, x1, y0, y1, kind):
    # center of the shape's geometry (bbox center for diamond; for star, top vertex + R)
    cx = (x0 + x1) / 2
    if kind == 'star':
        cy = y0 + (y1 - y0) / (1 + np.cos(np.radians(36)))
    else:
        cy = (y0 + y1) / 2
    return cx, cy

def kind_of(m):
    x0, x1, y0, y1 = bbox(m)
    fill = m.sum() / ((x1 - x0 + 1) * (y1 - y0 + 1))
    return 'diamond' if fill > 0.4 else 'star'

objects = []
for idx, _ in solids:
    m = lab == idx
    k = kind_of(m)
    x0, x1, y0, y1 = bbox(m)
    objects.append(dict(mask=m, kind=k, center=ref_center(x0, x1, y0, y1, k), color=dominant_color(m)))

# Outline pixels: all non-solid components; group by color (two outline colors).
outline_mask = mask.copy()
for idx, _ in solids:
    outline_mask &= lab != idx
ocols = np.unique(src[outline_mask], axis=0)
outlines = []
for c in ocols:
    m = outline_mask & np.all(src == c, axis=2)
    x0, x1, y0, y1 = bbox(m)
    outlines.append(dict(color=c, bbox=(x0, x1, y0, y1)))

# Match each object to the outline with nearest hue (same color family).
def match(obj):
    oc = obj['color'].astype(float)
    best = min(outlines, key=lambda o: np.linalg.norm(o['color'].astype(float) / max(o['color'].max(), 1) - oc / max(oc.max(), 1)))
    return best

background = src.copy()
for o in objects:
    background[o['mask']] = BG
    ys, xs = np.where(o['mask'])
    o['ys'], o['xs'], o['pix'] = ys, xs, src[ys, xs]
    ol = match(o)
    o['target'] = ref_center(*ol['bbox'], o['kind'])

def ease(t):  # smooth start/stop
    return t * t * (3 - 2 * t)

tmp = tempfile.mkdtemp()
for f in range(N_FRAMES):
    t = ease(f / (N_FRAMES - 1))
    frame = background.copy()
    for o in objects:
        dx = int(round((o['target'][0] - o['center'][0]) * t))
        dy = int(round((o['target'][1] - o['center'][1]) * t))
        ys = np.clip(o['ys'] + dy, 0, H - 1); xs = np.clip(o['xs'] + dx, 0, W - 1)
        frame[ys, xs] = o['pix']
    Image.fromarray(frame).save(f'{tmp}/f{f:03d}.png')

os.makedirs('/app/output', exist_ok=True)
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS), '-i', f'{tmp}/f%03d.png',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '12', '/app/output/video.mp4'], check=True)
Image.fromarray(frame).save('/app/output/last_frame.png')
print('done', [(o['kind'], o['center'], o['target']) for o in objects])
