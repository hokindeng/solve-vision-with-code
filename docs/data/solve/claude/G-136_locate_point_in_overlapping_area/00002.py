import numpy as np, cv2, subprocess, os
from PIL import Image

W = H = 1024; FPS = 16; N = 37
base = np.array(Image.open('/app/first_frame.png').convert('RGB'))

# --- detect the overlap region (unique blended colour) and the black points ---
ov_col = np.array([130, 196, 170])
ov = (np.abs(base.astype(int) - ov_col).sum(2) < 30).astype(np.uint8)
black = (base.sum(2) < 100).astype(np.uint8)
# fill point-holes inside the overlap region
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
ov_filled = cv2.morphologyEx(ov, cv2.MORPH_CLOSE, k)
ov_filled = cv2.erode(ov_filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))  # strict interior

n, lab, stats, cents = cv2.connectedComponentsWithStats(black)
inside = []
for i in range(1, n):
    if stats[i, cv2.CC_STAT_AREA] < 10: continue
    m = (lab == i).astype(np.uint8)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))  # margin: must be fully inside
    if np.all(ov_filled[m > 0] == 1):
        inside.append(tuple(cents[i]))
inside.sort(key=lambda p: (p[1], p[0]))
print('points inside overlap:', inside)

# --- animation: reveal red circles one at a time, each growing in ---
R = 18; RED = (220, 30, 30)
per = (N - 3) / max(len(inside), 1)
os.makedirs('/app/output', exist_ok=True)
frames = []
for f in range(N):
    img = base.copy()
    t = f  # frame index
    for j, (x, y) in enumerate(inside):
        start = 1 + j * per
        if t < start: continue
        prog = min(1.0, (t - start) / max(per * 0.7, 1))
        # sweep the circle around like being drawn
        ang = int(360 * prog)
        cx, cy = int(round(x)), int(round(y))
        if ang >= 360:
            cv2.circle(img, (cx, cy), R, RED, 3, cv2.LINE_AA)
        elif ang > 0:
            cv2.ellipse(img, (cx, cy), (R, R), -90, 0, ang, RED, 3, cv2.LINE_AA)
    frames.append(img)
frames[0] = base.copy()  # first frame identical to input

p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                      '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                      '-crf', '8', '-preset', 'slow', '/app/output/video.mp4'], stdin=subprocess.PIPE)
for fr in frames: p.stdin.write(fr.tobytes())
p.stdin.close(); p.wait()
print('wrote /app/output/video.mp4')
