import numpy as np, cv2, subprocess, os
from PIL import Image

W = H = 1024; FPS = 16; N = 80
src = np.array(Image.open('/app/first_frame.png').convert('RGB'))
bg = np.array([255, 255, 255], dtype=np.uint8)

# detect circles as connected non-background components
mask = (np.abs(src.astype(int) - 255).sum(2) > 0).astype(np.uint8)
n, lab, st, cen = cv2.connectedComponentsWithStats(mask, 8)
circles = []
for i in range(1, n):
    x, y, w, h, a = st[i]
    if a < 200: continue
    m = (lab[y:y+h, x:x+w] == i)
    circles.append(dict(x=x, y=y, w=w, h=h, m=m, patch=src[y:y+h, x:x+w].copy(), d=max(w, h)))

# target layout: sorted by circumference (i.e. diameter) descending, one row, centered
order = sorted(range(len(circles)), key=lambda k: -circles[k]['d'])
gap = 30
total = sum(circles[k]['w'] for k in order) + gap * (len(order) - 1)
cx = (W - total) / 2.0
for k in order:
    c = circles[k]
    c['tx'] = cx; c['ty'] = H / 2.0 - c['h'] / 2.0
    cx += c['w'] + gap

# background with circles removed
bgimg = src.copy()
for c in circles:
    bgimg[c['y']:c['y']+c['h'], c['x']:c['x']+c['w']][c['m']] = bg

def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)

os.makedirs('/app/output', exist_ok=True)
frames = []
for f in range(N):
    t = ease(min(1.0, f / (N - 1)))
    img = bgimg.copy()
    for k in order:  # draw larger circles first so smaller ones stay visible in transit
        c = circles[k]
        px = int(round(c['x'] + (c['tx'] - c['x']) * t))
        py = int(round(c['y'] + (c['ty'] - c['y']) * t))
        region = img[py:py+c['h'], px:px+c['w']]
        region[c['m']] = c['patch'][c['m']]
    frames.append(img)
frames[0] = src.copy()

p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                      '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                      '-crf', '10', '/app/output/video.mp4'], stdin=subprocess.PIPE)
for fr in frames: p.stdin.write(fr.tobytes())
p.stdin.close(); p.wait()
Image.fromarray(frames[-1]).save('/app/output/last_frame.png')
print('done', len(frames))
