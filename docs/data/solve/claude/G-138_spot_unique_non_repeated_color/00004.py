import numpy as np, cv2, subprocess, os
from PIL import Image

SRC = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
N_FRAMES, FPS, THICK = 21, 16, 6

img = np.array(Image.open(SRC).convert('RGB'))
H, W, _ = img.shape

# --- find unique color ---
flat = img.reshape(-1, 3)
cols, counts = np.unique(flat, axis=0, return_counts=True)
bg = tuple(cols[counts.argmax()])
shapes = []  # (color, mask)
for c, n in zip(cols, counts):
    c = tuple(c)
    if c == bg or n < 500:
        continue
    m = np.all(img == c, axis=2).astype(np.uint8)
    k, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    for i in range(1, k):
        if stats[i][4] > 500:
            shapes.append((c, (lab == i).astype(np.uint8)))
color_count = {}
for c, _ in shapes:
    color_count[c] = color_count.get(c, 0) + 1
uniq = [c for c, n in color_count.items() if n == 1]
assert len(uniq) == 1, color_count
target = [m for c, m in shapes if c == uniq[0]][0]

# --- contour path ---
cnts, _ = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2)
pts = np.vstack([cnt, cnt[:1]])
seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
cum = np.concatenate([[0], np.cumsum(seg)])
total = cum[-1]

def frame(t):
    f = img.copy()
    if t <= 0:
        return f
    L = total * t
    idx = np.searchsorted(cum, L)
    path = pts[:idx + 1]
    if idx < len(pts) and idx > 0:
        a, b = pts[idx - 1], pts[idx]
        r = (L - cum[idx - 1]) / max(seg[idx - 1], 1e-9)
        path = np.vstack([path[:-1], (a + r * (b - a))])
    path = path.round().astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(f, [path], isClosed=(t >= 1), color=(0, 0, 0),
                  thickness=THICK, lineType=cv2.LINE_8)
    return f

os.makedirs(os.path.dirname(OUT), exist_ok=True)
tmp = '/app/output/_frames'
os.makedirs(tmp, exist_ok=True)
for i in range(N_FRAMES):
    t = i / (N_FRAMES - 1)
    Image.fromarray(frame(t)).save(f'{tmp}/{i:03d}.png')
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS),
                '-i', f'{tmp}/%03d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-crf', '12', OUT], check=True)
for fn in os.listdir(tmp):
    os.remove(f'{tmp}/{fn}')
os.rmdir(tmp)
print('wrote', OUT)
