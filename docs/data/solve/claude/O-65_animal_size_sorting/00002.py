"""Animate animal faces sorting by size (largest -> smallest) onto the bottom baseline."""
import subprocess, os
import numpy as np
from PIL import Image
from scipy import ndimage

SRC = '/app/first_frame.png'
OUT_DIR = '/app/output'
OUT = os.path.join(OUT_DIR, 'video.mp4')
N_FRAMES, FPS = 40, 16

img = np.array(Image.open(SRC).convert('RGB'))
H, W, _ = img.shape
nonwhite = (img != 255).any(axis=2)

# --- locate the baseline (long horizontal run of non-white pixels) ---
row_counts = nonwhite.sum(axis=1)
base_rows = np.where(row_counts > W // 2)[0]
baseline_top = int(base_rows.min())
base_cols = np.where(nonwhite[baseline_top])[0]
base_x0, base_x1 = int(base_cols.min()), int(base_cols.max())

# --- find the animal faces as connected components above the baseline ---
mask = nonwhite.copy()
mask[baseline_top - 1:] = False
lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
boxes = []
for i, s in enumerate(ndimage.find_objects(lab)):
    boxes.append([s[1].start, s[0].start, s[1].stop, s[0].stop])  # x0,y0,x1,y1
# merge components nested inside another (e.g. pupils inside white eyes)
boxes.sort(key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
merged = []
for b in boxes:
    for m in merged:
        if b[0] >= m[0] and b[1] >= m[1] and b[2] <= m[2] and b[3] <= m[3]:
            break
    else:
        merged.append(b)

sprites = []
for x0, y0, x1, y1 in merged:
    patch = img[y0:y1, x0:x1].copy()
    alpha = nonwhite[y0:y1, x0:x1].copy()
    sprites.append(dict(x=x0, y=y0, w=x1 - x0, h=y1 - y0, rgb=patch, a=alpha))

# background = frame with sprites removed (everything else untouched)
bg = img.copy()
for s in sprites:
    region = bg[s['y']:s['y'] + s['h'], s['x']:s['x'] + s['w']]
    region[s['a']] = 255

# --- targets: sort by size (largest first), evenly spaced along the baseline ---
order = sorted(range(len(sprites)), key=lambda i: -(sprites[i]['w'] * sprites[i]['h']))
total_w = sum(sprites[i]['w'] for i in order)
gap = (base_x1 - base_x0 + 1 - total_w) / (len(order) + 1)
cx = base_x0 + gap
for i in order:
    s = sprites[i]
    s['tx'] = int(round(cx))
    s['ty'] = baseline_top - s['h']          # bottom edge rests on the baseline
    cx += s['w'] + gap

def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)

MOVE_START, MOVE_END = 1, 35  # hold first frame, settle for last frames

os.makedirs(OUT_DIR, exist_ok=True)
frames = []
for f in range(N_FRAMES):
    t = np.clip((f - MOVE_START) / (MOVE_END - MOVE_START), 0, 1)
    e = ease(t)
    frame = bg.copy()
    # draw in size order so larger ones are behind smaller ones if they cross
    for i in order:
        s = sprites[i]
        px = int(round(s['x'] + (s['tx'] - s['x']) * e))
        py = int(round(s['y'] + (s['ty'] - s['y']) * e))
        region = frame[py:py + s['h'], px:px + s['w']]
        region[s['a']] = s['rgb'][s['a']]
    frames.append(frame)

# sanity: first frame identical to the source
assert (frames[0] == img).all()

cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
       '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
       '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '15', '-r', str(FPS), OUT]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for fr in frames:
    p.stdin.write(fr.tobytes())
p.stdin.close()
p.wait()
print('wrote', OUT, 'sprites:', [(s['w'], s['h'], s['tx'], s['ty']) for s in (sprites[i] for i in order)])
