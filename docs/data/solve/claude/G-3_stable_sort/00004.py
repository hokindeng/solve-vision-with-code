import numpy as np, cv2, subprocess, os
from PIL import Image

W = H = 1024; FPS = 16; N = 96
src = np.array(Image.open('/app/first_frame.png').convert('RGB'))
bg_color = np.array([235, 235, 235])
mask = (np.abs(src.astype(int) - bg_color).sum(2) > 10).astype(np.uint8)
n, lab, st, cen = cv2.connectedComponentsWithStats(mask)

# clean background: fill shapes with background color
bg = src.copy(); bg[mask > 0] = bg_color

shapes = []
for i in range(1, n):
    x, y, w, h, a = st[i]
    m = (lab == i)
    fill = w * h
    kind = 'square' if a / fill > 0.9 else 'triangle'
    shapes.append(dict(x=int(x), y=int(y), w=int(w), h=int(h), size=int(w), kind=kind,
                       pix=src[y:y+h, x:x+w].copy(), m=m[y:y+h, x:x+w].copy(),
                       cx=cen[i][0]))

# group order: by mean x of each type in the original frame
kinds = sorted({s['kind'] for s in shapes},
               key=lambda k: np.mean([s['cx'] for s in shapes if s['kind'] == k]))
ordered = []
for k in kinds:
    ordered += sorted([s for s in shapes if s['kind'] == k], key=lambda s: s['size'])

# target layout: single horizontal line, vertically centered
total = sum(s['w'] for s in ordered); gap = 40
x0 = (W - (total + gap * (len(ordered) - 1))) // 2
cy = H // 2
for s in ordered:
    s['tx'] = x0; s['ty'] = cy - s['h'] // 2
    x0 += s['w'] + gap

def ease(t):
    return t * t * (3 - 2 * t)

hold0, hold1 = 6, 8
move = N - hold0 - hold1
os.makedirs('/app/output', exist_ok=True)
ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                       '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264',
                       '-pix_fmt', 'yuv420p', '-crf', '15', '/app/output/video.mp4'], stdin=subprocess.PIPE)
for f in range(N):
    t = min(max((f - hold0) / (move - 1), 0.0), 1.0)
    e = ease(t)
    frame = bg.copy()
    # draw larger shapes first so smaller ones stay visible if paths cross
    for s in sorted(ordered, key=lambda s: -s['size']):
        x = int(round(s['x'] + (s['tx'] - s['x']) * e))
        y = int(round(s['y'] + (s['ty'] - s['y']) * e))
        reg = frame[y:y+s['h'], x:x+s['w']]
        reg[s['m']] = s['pix'][s['m']]
    ff.stdin.write(frame.tobytes())
ff.stdin.close(); ff.wait()
