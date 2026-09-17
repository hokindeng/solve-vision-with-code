"""Circle the single asymmetrical shape in first_frame.png with an animated red circle."""
import numpy as np, cv2, subprocess, os, math
from PIL import Image

SRC = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
N, FPS = 16, 16

img = np.array(Image.open(SRC).convert('RGB'))
H, W = img.shape[:2]
bg = img[0, 0]
diff = np.abs(img.astype(int) - bg.astype(int)).sum(2)
mask = (diff > 40).astype(np.uint8)
n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)


def asym_score(i):
    """Best-axis mirror mismatch (soft, 4x upscaled) normalised by perimeter.
    Symmetric shapes give ~1.5-2 (discretisation only); asymmetric ones much more."""
    x, y, w, h, _ = stats[i]
    p = 4
    sl = (slice(max(y - p, 0), y + h + p), slice(max(x - p, 0), x + w + p))
    own = (lab[sl] == i)
    soft = (np.clip(diff[sl] / diff[lab == i].max(), 0, 1) * own).astype(np.float32)
    m = cv2.resize(soft, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    M = cv2.moments(m)
    cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
    cnts, _ = cv2.findContours((m > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    per = max(cv2.arcLength(c, True) for c in cnts)
    best = 1e18
    for k in range(360):
        a = math.radians(k / 2)
        c, s = math.cos(2 * a), math.sin(2 * a)
        A = np.array([[c, s, cx - c * cx - s * cy], [s, -c, cy - s * cx + c * cy]])
        r = cv2.warpAffine(m, A, (m.shape[1], m.shape[0]), flags=cv2.INTER_LINEAR)
        best = min(best, np.abs(m - r).sum())
    return best / per


cands = sorted(((asym_score(i), i) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 50), reverse=True)
target = cands[0][1]
x, y, w, h = stats[target, :4]
cx, cy = x + w / 2, y + h / 2
r = int(math.hypot(w, h) / 2 + 22)

os.makedirs('/app/output', exist_ok=True)
tmp = '/app/output/frames'
os.makedirs(tmp, exist_ok=True)
for f in range(N):
    fr = img.copy()
    if f > 0:
        sweep = 360 * f / (N - 1)          # circle draws progressively; full at last frame
        cv2.ellipse(fr, (int(round(cx)), int(round(cy))), (r, r), -90, 0, sweep,
                    (255, 0, 0), 6, cv2.LINE_AA)
    Image.fromarray(fr).save(f'{tmp}/{f:03d}.png')

subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS),
                '-i', f'{tmp}/%03d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-crf', '12', OUT], check=True)
for fn in os.listdir(tmp):
    os.remove(f'{tmp}/{fn}')
os.rmdir(tmp)
print('scores', [(round(s, 2), i) for s, i in cands], '-> target', target, 'center', (cx, cy), 'radius', r)
