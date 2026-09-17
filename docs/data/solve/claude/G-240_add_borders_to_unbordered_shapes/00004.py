import subprocess
import numpy as np
import cv2
from PIL import Image

W = H = 1024
FPS = 16
N_FRAMES = 80
BORDER_W = 5          # thickness of existing border in first_frame (measured)
BG = (255, 255, 255)
BLACK = (0, 0, 0)

src = np.array(Image.open('/app/first_frame.png').convert('RGB'))

# --- find shapes: connected components of non-background, non-black color ---
nonbg = np.any(src != np.array(BG, np.uint8), axis=2)
black = np.all(src == np.array(BLACK, np.uint8), axis=2)
fill = (nonbg & ~black).astype(np.uint8)
n, lab, stats, cents = cv2.connectedComponentsWithStats(fill, connectivity=8)

rings = []   # (ring_mask, cx, cy)
for i in range(1, n):
    if stats[i][4] < 50:
        continue
    m = (lab == i)
    # does it already have a black border? check for black pixels just outside the fill
    dil = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) .astype(bool) & ~m
    if black[dil].mean() > 0.5:
        continue  # already bordered
    # inside border ring: fill pixels within BORDER_W of the outside
    dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    ring = m & (dist <= BORDER_W)
    cx, cy = cents[i]
    rings.append((ring, cx, cy))

# order shapes left-to-right / top-to-bottom for a deterministic sequence
rings.sort(key=lambda r: (r[2] // 200, r[1]))

# --- animation schedule ---
hold_start, hold_end = 6, 6
active = N_FRAMES - hold_start - hold_end
per = active / max(1, len(rings))

yy, xx = np.mgrid[0:H, 0:W]

def ease(t):
    return t * t * (3 - 2 * t)

def frame_at(f):
    img = src.copy()
    for k, (ring, cx, cy) in enumerate(rings):
        t0 = hold_start + k * per
        p = np.clip((f - t0) / per, 0, 1)
        if p <= 0:
            continue
        p = ease(p)
        # angular sweep, clockwise from top
        ang = (np.arctan2(xx - cx, -(yy - cy)) + 2 * np.pi) % (2 * np.pi)
        sel = ring & (ang <= p * 2 * np.pi + 1e-9)
        img[sel] = BLACK
    return img

proc = subprocess.Popen([
    'ffmpeg', '-y', '-loglevel', 'error',
    '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '15', '-preset', 'medium',
    '-movflags', '+faststart', '/app/output/video.mp4'], stdin=subprocess.PIPE)
for f in range(N_FRAMES):
    proc.stdin.write(frame_at(f).tobytes())
proc.stdin.close()
proc.wait()
Image.fromarray(frame_at(N_FRAMES - 1)).save('/app/output/last_frame.png')
print('shapes bordered:', len(rings))
