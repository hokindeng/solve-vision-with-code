import numpy as np, cv2, subprocess, os
from PIL import Image

SRC = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
FPS, N = 16, 48

base = np.array(Image.open(SRC).convert('RGB'))
H, W, _ = base.shape

# --- detect lines: connected components of non-background pixels ---
bg = base[0, 0]
mask = (np.abs(base.astype(int) - bg.astype(int)).sum(2) > 30).astype(np.uint8)
n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
horizontal = []
for i in range(1, n):
    x, y, w, h, area = stats[i]
    if area < 50:
        continue
    if w > 3 * h:  # wide and thin -> horizontal
        horizontal.append((x, y, w, h))

# circle params per horizontal line
circles = []
for x, y, w, h in horizontal:
    cx, cy = x + w / 2.0, y + h / 2.0
    r = int(w / 2.0 + 18)
    circles.append((cx, cy, r))

# --- animation: arc sweeps 0->360 deg over frames 1..DRAW_END, then hold ---
DRAW_START, DRAW_END = 2, 42
SS = 4  # supersampling for smooth edges
THICK = 5

def frame(t):
    img = base.copy()
    if t < DRAW_START:
        return img
    p = min(1.0, (t - DRAW_START) / float(DRAW_END - DRAW_START))
    sweep = 360.0 * p
    if sweep <= 0:
        return img
    for cx, cy, r in circles:
        # draw on a supersampled patch and blend down for anti-aliasing
        pad = r + THICK + 4
        x0, y0 = int(cx - pad), int(cy - pad)
        x1, y1 = int(cx + pad) + 1, int(cy + pad) + 1
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        pw, ph = x1 - x0, y1 - y0
        layer = np.zeros((ph * SS, pw * SS), np.uint8)
        c = (int(round((cx - x0) * SS)), int(round((cy - y0) * SS)))
        cv2.ellipse(layer, c, (r * SS, r * SS), 0, -90, -90 + sweep,
                    255, THICK * SS, lineType=cv2.LINE_AA)
        alpha = cv2.resize(layer, (pw, ph), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        patch = img[y0:y1, x0:x1].astype(np.float32)
        patch = patch * (1 - alpha[..., None])  # black stroke
        img[y0:y1, x0:x1] = np.clip(patch, 0, 255).astype(np.uint8)
    return img

frames = [frame(t) for t in range(N)]
assert np.array_equal(frames[0], base)

tmp = '/app/output/_frames'
os.makedirs(tmp, exist_ok=True)
for i, f in enumerate(frames):
    Image.fromarray(f).save(f'{tmp}/{i:04d}.png')
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS),
                '-i', f'{tmp}/%04d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-crf', '12', '-preset', 'slow', OUT], check=True)
for fn in os.listdir(tmp):
    os.remove(os.path.join(tmp, fn))
os.rmdir(tmp)
print('horizontal lines:', horizontal, 'circles:', circles)
