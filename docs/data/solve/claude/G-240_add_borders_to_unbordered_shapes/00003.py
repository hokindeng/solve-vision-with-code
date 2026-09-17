import numpy as np, cv2, subprocess, os
from PIL import Image

SRC = '/app/first_frame.png'; OUT = '/app/output/video.mp4'
W = H = 1024; FPS = 16; N = 80; BORDER = 5

im = np.array(Image.open(SRC).convert('RGB'))
nonwhite = (im != 255).any(2).astype(np.uint8)
black = (im == 0).all(2)
n, lab, stats, cent = cv2.connectedComponentsWithStats(nonwhite, 8)

# Build the full border mask for every shape lacking black border pixels.
border_mask = np.zeros((H, W), bool)
angle = np.zeros((H, W), np.float32)
for i in range(1, n):
    m = lab == i
    if black[m].mean() > 0.02:      # already bordered
        continue
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * BORDER + 1, 2 * BORDER + 1))
    inner = cv2.erode(m.astype(np.uint8), k).astype(bool)
    ring = m & ~inner
    cy, cx = cent[i][1], cent[i][0]
    ys, xs = np.nonzero(ring)
    a = (np.arctan2(ys - cy, xs - cx) + np.pi / 2) % (2 * np.pi)  # start at top, clockwise
    angle[ys, xs] = a
    border_mask |= ring

os.makedirs(os.path.dirname(OUT), exist_ok=True)
ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                       '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264',
                       '-pix_fmt', 'yuv420p', '-crf', '12', '-preset', 'slow', OUT], stdin=subprocess.PIPE)
for f in range(N):
    t = f / (N - 1)
    frame = im.copy()
    if f > 0:
        thr = 2 * np.pi * min(1.0, t * 1.0)
        sel = border_mask & (angle <= thr) if f < N - 1 else border_mask
        frame[sel] = 0
    ff.stdin.write(frame.tobytes())
ff.stdin.close(); ff.wait()
print('wrote', OUT)
