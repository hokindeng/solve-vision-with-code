"""Draw a red circle around the leftmost shape, animated as an arc sweep over 3 s."""
import numpy as np, cv2, subprocess, os

W = H = 1024; FPS = 16; N = 48
BASE = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
TMP = '/app/output/frames'

img = cv2.imread(BASE)
# find shapes as connected components of non-background pixels
mask = (np.abs(img.astype(int) - img[0, 0].astype(int)).sum(2) > 30).astype(np.uint8)
n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
comps = [(stats[i], cent[i]) for i in range(1, n) if stats[i][4] > 200]
st, c = min(comps, key=lambda t: t[0][0])          # leftmost = smallest bbox x
x, y, w, h, _ = st
cx, cy = x + w / 2.0, y + h / 2.0
radius = int(np.hypot(w, h) / 2 + 22)              # encloses bbox with margin
color = (0, 0, 255)                                # red (BGR)
thick = 6

os.makedirs(TMP, exist_ok=True)
for f in os.listdir(TMP): os.remove(os.path.join(TMP, f))
for i in range(N):
    fr = img.copy()
    t = i / (N - 1)
    sweep = 360.0 * min(1.0, t * 1.08)               # complete slightly before end
    if i > 0 and sweep > 0:
        cv2.ellipse(fr, (int(round(cx)), int(round(cy))), (radius, radius),
                    0, -90, -90 + sweep, color, thick, cv2.LINE_AA)
    cv2.imwrite(f'{TMP}/{i:03d}.png', fr)

subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS),
                '-i', f'{TMP}/%03d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-crf', '12', '-r', str(FPS), OUT], check=True)
print('wrote', OUT, 'circle at', (cx, cy), 'r', radius)
