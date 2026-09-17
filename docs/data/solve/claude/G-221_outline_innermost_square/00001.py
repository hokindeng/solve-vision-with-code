"""Outline the innermost concentric square with a blue outline, step by step."""
import subprocess, numpy as np
from PIL import Image

FRAMES, FPS, SIZE = 85, 16, 1024
BLUE = np.array([0, 0, 255], dtype=np.uint8)
THICK = 8  # outline thickness (pixels), centred on the square's edge

base = np.array(Image.open('/app/first_frame.png').convert('RGB'))
bg = tuple(base[0, 0])

# Find concentric squares: each non-background colour is one square; innermost = smallest bbox.
cols = {tuple(c) for c in np.unique(base.reshape(-1, 3), axis=0)} - {bg}
boxes = []
for c in cols:
    ys, xs = np.nonzero(np.all(base == c, axis=2))
    boxes.append((xs.max() - xs.min(), xs.min(), ys.min(), xs.max(), ys.max()))
_, x0, y0, x1, y1 = min(boxes)
h = THICK // 2
L, T, R, B = x0 - h, y0 - h, x1 + h + 1, y1 + h + 1  # outer bounds of the outline band
perim = 2 * ((R - L) + (B - T))

def draw_outline(img, frac):
    """Draw the first `frac` of the outline perimeter, going top->right->bottom->left."""
    n = frac * perim
    W, H = R - L, B - T
    segs = [  # (length, draw fn(t) where t = pixels drawn along this side)
        (W, lambda t: img.__setitem__((slice(T, T + THICK), slice(L, L + t)), BLUE)),
        (H, lambda t: img.__setitem__((slice(T, T + t), slice(R - THICK, R)), BLUE)),
        (W, lambda t: img.__setitem__((slice(B - THICK, B), slice(R - t, R)), BLUE)),
        (H, lambda t: img.__setitem__((slice(B - t, B), slice(L, L + THICK)), BLUE)),
    ]
    for length, fn in segs:
        t = int(round(min(n, length)))
        if t > 0:
            fn(t)
        n -= length
        if n <= 0:
            break

ANIM_START, ANIM_END = 4, FRAMES - 10  # hold at start, draw, then hold completed result
frames = []
for i in range(FRAMES):
    img = base.copy()
    if i >= ANIM_START:
        frac = min(1.0, (i - ANIM_START) / (ANIM_END - ANIM_START))
        draw_outline(img, frac)
    frames.append(img)

cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
       '-s', f'{SIZE}x{SIZE}', '-r', str(FPS), '-i', '-',
       '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '12', '/app/output/video.mp4']
p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for f in frames:
    p.stdin.write(f.tobytes())
p.stdin.close(); p.wait()
print('wrote /app/output/video.mp4', p.returncode)
