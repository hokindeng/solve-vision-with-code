import numpy as np, cv2, subprocess, os
from PIL import Image

SRC = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
N_FRAMES, FPS = 40, 16
RED = (255, 0, 0)

img = np.array(Image.open(SRC).convert('RGB'))
H, W = img.shape[:2]

# --- find the topmost (unobscured) shape ---
# Each colored region: the topmost shape is the one whose pixel mask is a
# single convex polygon that no other shape's region intrudes into. Equivalently,
# it is the only shape whose observed region equals its convex hull.
cols, counts = np.unique(img.reshape(-1, 3), axis=0, return_counts=True)
best, best_score = None, -1
for c, n in zip(cols, counts):
    if n < 500 or tuple(c) == (255, 255, 255):
        continue
    mask = np.all(img == c, axis=2).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    score = area / hull_area  # 1.0 => unobscured convex shape
    if score > best_score:
        best_score, best, best_cnt = score, c, cnt

# simplify to polygon vertices
peri = cv2.arcLength(best_cnt, True)
poly = cv2.approxPolyDP(best_cnt, 0.01 * peri, True).reshape(-1, 2).astype(np.float64)

# outline path (closed), cumulative lengths
pts = np.vstack([poly, poly[:1]])
seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
cum = np.concatenate([[0], np.cumsum(seg)])
total = cum[-1]

def partial_path(frac):
    L = frac * total
    out = [pts[0]]
    for i in range(len(seg)):
        if cum[i + 1] <= L:
            out.append(pts[i + 1])
        else:
            t = (L - cum[i]) / seg[i] if seg[i] > 0 else 0
            out.append(pts[i] + t * (pts[i + 1] - pts[i]))
            break
    return np.array(out)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
frames_dir = '/app/output/frames'
os.makedirs(frames_dir, exist_ok=True)
thick = 6
for f in range(N_FRAMES):
    frame = img.copy()
    if f > 0:
        frac = min(1.0, f / (N_FRAMES - 1 - 4))  # finish a few frames early, hold at end
        p = partial_path(frac)
        closed = frac >= 1.0
        cv2.polylines(frame, [np.round(p).astype(np.int32)], closed, RED, thick, cv2.LINE_AA)
    Image.fromarray(frame).save(f'{frames_dir}/{f:03d}.png')

subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS),
                '-i', f'{frames_dir}/%03d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-crf', '18', OUT], check=True)
print('topmost color', best, 'score', best_score, 'vertices', len(poly))
