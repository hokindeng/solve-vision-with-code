import numpy as np, cv2, subprocess, os
from PIL import Image

SRC = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
N_FRAMES, FPS = 35, 16

img = np.array(Image.open(SRC).convert('RGB'))
H, W, _ = img.shape
bg = np.array([220, 220, 220], np.uint8)
nonbg = np.any(img != bg, axis=2)

# Split foreground by color family (olive-ish vs magenta-ish)
r, g, b = [img[..., i].astype(int) for i in range(3)]
families = {
    'olive': nonbg & (abs(r - g) < 10) & (b < r),
    'magenta': nonbg & (r > 150) & (b > 150) & (g < 100),
}

def components(mask):
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [(lab == i, stats[i], cents[i]) for i in range(1, n)]

moves = []  # (object mask, dx, dy)
for name, fam in families.items():
    comps = components(fam)
    comps.sort(key=lambda c: -c[1][cv2.CC_STAT_AREA])
    obj_mask = comps[0][0]
    # fill holes (e.g. highlight dot inside circle) so object mask is solid
    obj_u8 = obj_mask.astype(np.uint8)
    cnts, _ = cv2.findContours(obj_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    obj_mask = cv2.drawContours(np.zeros_like(obj_u8), cnts, -1, 1, -1).astype(bool)
    # outline = remaining pixels of this family not in object
    outline = fam & ~obj_mask
    ys, xs = np.where(outline)
    tgt = np.array([(xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2])
    ys, xs = np.where(obj_mask)
    src = np.array([(xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2])
    moves.append((obj_mask, tgt - src))
    print(name, 'from', src, 'to', tgt)

# base frame: background with objects erased (outlines and everything else preserved)
base = img.copy()
for m, _ in moves:
    base[m] = bg

sprites = []
for m, d in moves:
    ys, xs = np.where(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sprites.append((img[y0:y1, x0:x1].copy(), m[y0:y1, x0:x1], x0, y0, d))

def ease(t):
    return t * t * (3 - 2 * t)  # smoothstep

frames = []
for i in range(N_FRAMES):
    t = ease(i / (N_FRAMES - 1))
    f = base.copy()
    for spr, sm, x0, y0, d in sprites:
        ox, oy = int(round(x0 + d[0] * t)), int(round(y0 + d[1] * t))
        h, w = sm.shape
        sub = f[oy:oy + h, ox:ox + w]
        sub[sm] = spr[sm]
    frames.append(f)
frames[0] = img.copy()

tmp = '/app/output/frames'
os.makedirs(tmp, exist_ok=True)
for i, f in enumerate(frames):
    Image.fromarray(f).save(f'{tmp}/{i:03d}.png')
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS), '-i', f'{tmp}/%03d.png',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '12', OUT], check=True)
print('wrote', OUT)
