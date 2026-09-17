import numpy as np, cv2, subprocess, os, math
from PIL import Image

BASE = np.array(Image.open('/app/first_frame.png').convert('RGB'))
H, W, _ = BASE.shape
N_FRAMES, FPS = 22, 16
os.makedirs('/app/output', exist_ok=True)

# --- Step 1: locate triangle vertices from the black stroke ---
mask = (BASE.sum(axis=2) < 300).astype(np.uint8) * 255
cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnt = max(cnts, key=cv2.contourArea)
hull = cv2.convexHull(cnt)
peri = cv2.arcLength(hull, True)
eps = 0.01
while True:
    approx = cv2.approxPolyDP(hull, eps * peri, True)
    if len(approx) <= 3: break
    eps += 0.005
verts = approx.reshape(-1, 2).astype(float)
assert len(verts) == 3, verts

# --- Step 2: interior angles ---
def angle_at(i):
    a, b, c = verts[i], verts[(i+1)%3], verts[(i+2)%3]
    u, v = b - a, c - a
    return math.degrees(math.acos(np.dot(u, v) / (np.linalg.norm(u)*np.linalg.norm(v))))
angles = [angle_at(i) for i in range(3)]
big = int(np.argmax(angles))
cx, cy = verts[big]
print('vertices:', verts.tolist()); print('angles:', angles, '-> largest at vertex', big)

# --- Step 3: animate a red circle sweeping around that vertex ---
RADIUS, THICK = 42, 5
RED = (220, 30, 30)

def frame(t):  # t in [0,1]
    img = BASE.copy()
    if t <= 0: return img
    over = img.copy()
    sweep = 360.0 * min(1.0, t)
    start = -90.0
    # draw arc with anti-aliasing via supersampling
    S = 4
    big_img = cv2.resize(over, (W*S, H*S), interpolation=cv2.INTER_NEAREST)
    cv2.ellipse(big_img, (int(round(cx*S)), int(round(cy*S))), (RADIUS*S, RADIUS*S), 0,
                start, start + sweep, RED, THICK*S, lineType=cv2.LINE_AA)
    small = cv2.resize(big_img, (W, H), interpolation=cv2.INTER_AREA)
    # only touch pixels inside the circle annulus region (keep everything else exact)
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.hypot(xx - cx, yy - cy)
    ring = (d > RADIUS - THICK/2 - 2) & (d < RADIUS + THICK/2 + 2)
    img[ring] = small[ring]
    return img

frames = []
for k in range(N_FRAMES):
    # frame 0 = original; sweep over frames 1..N-3; hold the finished circle at the end
    t = 0.0 if k == 0 else min(1.0, (k) / (N_FRAMES - 3))
    frames.append(frame(t))

tmp = '/app/output/frames'; os.makedirs(tmp, exist_ok=True)
for k, f in enumerate(frames):
    Image.fromarray(f).save(f'{tmp}/f{k:03d}.png')
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS), '-i', f'{tmp}/f%03d.png',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '12', '-preset', 'slow',
                '/app/output/video.mp4'], check=True)
for fn in os.listdir(tmp): os.remove(os.path.join(tmp, fn))
os.rmdir(tmp)
print('wrote /app/output/video.mp4')
