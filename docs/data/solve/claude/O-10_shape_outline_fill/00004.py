import numpy as np, subprocess, os
from PIL import Image

FIRST = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
N, FPS = 60, 16

base = np.array(Image.open(FIRST).convert('RGB')).astype(np.float32)
H, W, _ = base.shape

# --- measured from first frame ---
GREEN = np.array([70, 153, 53], np.float32)
CX, CY = 864, 768          # answer slot centre (matches column of top-right ellipse)
R_OUT, R_IN = 80.0, 76.0   # circle C: outer radius 80, 4px stroke
DASH_PERIOD, GAP_FULL = 8.0, 2.0   # top-right ellipse: 6 on / 2 off
QBOX = (830, 730, 900, 810)  # region holding the '?' (x0,y0,x1,y1)

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
dx, dy = xx - CX, yy - CY
rr = np.sqrt(dx*dx + dy*dy)
theta = np.mod(np.arctan2(dy, dx) + np.pi/2, 2*np.pi)  # start at top, clockwise
ring = (rr >= R_IN) & (rr < R_OUT)
arclen = R_OUT * theta

# region without the '?': white background there
clean = base.copy()
x0, y0, x1, y1 = QBOX
clean[y0:y1, x0:x1] = 255

def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t*t*(3-2*t)

def frame(i):
    t = i / (N - 1)
    img = base.copy()
    # phase 1: '?' fades out (t 0 -> 0.25)
    a = ease(t / 0.25)
    img = (1-a)*base + a*clean
    # phase 2: solid circle sweeps in (t 0.2 -> 0.6)
    sweep = ease((t - 0.2) / 0.4)
    # phase 3: gaps open up -> dashed (t 0.6 -> 1.0)
    gap = GAP_FULL * ease((t - 0.6) / 0.4)
    if sweep > 0:
        m = ring & (theta <= sweep * 2*np.pi + 1e-6)
        if gap > 0:
            m &= (np.mod(arclen, DASH_PERIOD) < DASH_PERIOD - gap)
        img[m] = GREEN
    return np.clip(img + 0.5, 0, 255).astype(np.uint8)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                      '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264',
                      '-pix_fmt', 'yuv420p', '-crf', '12', OUT], stdin=subprocess.PIPE)
for i in range(N):
    f = frame(i)
    if i == 0:
        assert np.array_equal(f, base.astype(np.uint8))
    if i == N-1:
        Image.fromarray(f).save('/app/output/last_frame.png')
    p.stdin.write(f.tobytes())
p.stdin.close(); p.wait()
print('wrote', OUT)
