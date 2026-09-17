import numpy as np, cv2, subprocess, os
from PIL import Image

SRC = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
N_FRAMES, FPS = 40, 16

img = np.array(Image.open(SRC).convert('RGB'))
H, W, _ = img.shape
nonwhite = (img != 255).any(2)

# baseline: long horizontal line rows
line_rows = np.where(nonwhite.sum(1) > W // 2)[0]
baseline_y = int(line_rows.min())
line_x = np.where(nonwhite[line_rows[0]])[0]
x_left, x_right = int(line_x.min()), int(line_x.max())

# components (excluding the line); merge boxes nested inside others (e.g. pupils)
mask = nonwhite.copy()
mask[line_rows] = False
n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
boxes = [tuple(stats[i, :4]) for i in range(1, n) if stats[i, 4] > 20]
def inside(b, o):
    return b[0] >= o[0] and b[1] >= o[1] and b[0]+b[2] <= o[0]+o[2] and b[1]+b[3] <= o[1]+o[3]
boxes = [b for b in boxes if not any(o is not b and inside(b, o) and o[4 if False else 2]*o[3] > b[2]*b[3] for o in boxes)]

sprites = []
for (x, y, w, h) in boxes:
    crop = img[y:y+h, x:x+w].copy()
    m = nonwhite[y:y+h, x:x+w].astype(np.uint8)
    # fill interior holes (e.g. white eyes) so they travel with the face
    ff = np.pad(m, 1)
    inv = (1 - ff).astype(np.uint8)
    flood = inv.copy()
    cv2.floodFill(flood, None, (0, 0), 2)
    holes = (flood == 1)[1:-1, 1:-1]
    alpha = (m | holes).astype(bool)
    sprites.append(dict(x=int(x), y=int(y), w=int(w), h=int(h), rgb=crop, a=alpha, size=int(alpha.sum())))

# sort largest -> smallest, targets along the baseline, bottoms on the line
order = sorted(range(len(sprites)), key=lambda i: -sprites[i]['size'])
total_w = sum(sprites[i]['w'] for i in order)
gap = (x_right - x_left - total_w) / (len(order) + 1)
cx = x_left + gap
for i in order:
    s = sprites[i]
    s['tx'] = int(round(cx)); s['ty'] = baseline_y - 1 - s['h']
    cx += s['w'] + gap

background = np.full_like(img, 255)
background[line_rows] = img[line_rows]

def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
tmp = '/app/output/_frames'
os.makedirs(tmp, exist_ok=True)
for f in range(N_FRAMES):
    t = ease(f / (N_FRAMES - 1))
    frame = background.copy()
    # draw larger first so smaller ones stay visible when paths cross
    for i in order:
        s = sprites[i]
        px = int(round(s['x'] + (s['tx'] - s['x']) * t))
        py = int(round(s['y'] + (s['ty'] - s['y']) * t))
        region = frame[py:py+s['h'], px:px+s['w']]
        region[s['a']] = s['rgb'][s['a']]
    Image.fromarray(frame).save(f'{tmp}/{f:03d}.png')

subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS), '-i', f'{tmp}/%03d.png',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '12', OUT], check=True)
import shutil; shutil.rmtree(tmp)
print('wrote', OUT)
