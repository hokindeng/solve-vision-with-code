import numpy as np, cv2, subprocess, os
from PIL import Image
from scipy import ndimage

SRC = '/app/first_frame.png'
OUT_DIR = '/app/output'
OUT = os.path.join(OUT_DIR, 'video.mp4')
FPS, N_FRAMES, W = 16, 48, 1024

base = np.array(Image.open(SRC).convert('RGB'))
protect = np.any(base != 255, axis=2)  # every non-background pixel stays untouched

# --- find shapes grouped by exact color
flat = base.reshape(-1, 3)
cols, counts = np.unique(flat, axis=0, return_counts=True)
colors = [tuple(int(v) for v in c) for c, n in zip(cols, counts) if n > 500 and tuple(c) != (255, 255, 255)]

def shapes_of(color):
    m = np.all(base == color, axis=2)
    lab, n = ndimage.label(m)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if len(xs) < 300:
            continue
        out.append(dict(mask=lab == i, cx=xs.mean(), cy=ys.mean(), x0=xs.min(), x1=xs.max()))
    return sorted(out, key=lambda s: s['cx'])

def bezier(p0, p1, p2, n=400):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2

def clip_to_boundary(pts, mask_a, mask_b):
    """Trim the polyline so it starts/ends just outside the two shapes."""
    def inside(m, p):
        x, y = int(round(p[0])), int(round(p[1]))
        return 0 <= x < W and 0 <= y < W and m[y, x]
    i = 0
    while i < len(pts) and inside(mask_a, pts[i]):
        i += 1
    j = len(pts) - 1
    while j > i and inside(mask_b, pts[j]):
        j -= 1
    return pts[max(i - 2, 0):j + 3]

# --- build one curve per color: leftmost -> rightmost shape, bowed away from the other color
mean_y_all = np.mean([s['cy'] for c in colors for s in shapes_of(c)])
curves = []
for color in colors:
    sh = shapes_of(color)
    a, b = sh[0], sh[-1]
    p0 = np.array([a['cx'], a['cy']]); p2 = np.array([b['cx'], b['cy']])
    mid = (p0 + p2) / 2
    direction = -1.0 if mid[1] < mean_y_all else 1.0   # top pair bows up, bottom pair bows down
    ctrl = mid + np.array([0.0, direction * 260.0])
    pts = clip_to_boundary(bezier(p0, ctrl, p2), a['mask'], b['mask'])
    curves.append((color, pts))
curves.sort(key=lambda c: c[1][:, 1].mean())  # draw top curve first

def render(progress):
    """progress: list of fractions [0..1] per curve."""
    img = base.copy()
    for (color, pts), f in zip(curves, progress):
        k = int(round(f * (len(pts) - 1)))
        if k >= 1:
            seg = np.round(pts[:k + 1]).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [seg], False, color, thickness=6, lineType=cv2.LINE_AA)
    img[protect] = base[protect]
    return img

os.makedirs(OUT_DIR, exist_ok=True)
per = (N_FRAMES - 1) // len(curves)  # frames allotted per curve
frames = []
for i in range(N_FRAMES):
    prog = []
    for c in range(len(curves)):
        start = 1 + c * per
        end = start + per - 1 if c < len(curves) - 1 else N_FRAMES - 1
        prog.append(float(np.clip((i - start + 1) / (end - start + 1), 0, 1)))
    frames.append(render(prog))

cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{W}', '-r', str(FPS),
       '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '15', '-preset', 'slow', OUT]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
for fr in frames:
    p.stdin.write(fr.tobytes())
p.stdin.close(); p.wait()
Image.fromarray(frames[-1]).save(os.path.join(OUT_DIR, 'last_frame.png'))
print('wrote', OUT, len(frames), 'frames')
