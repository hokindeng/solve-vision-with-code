import numpy as np, subprocess, os
from PIL import Image

SRC = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
N, FPS = 25, 16
BOX_RGB = np.array([70, 140, 70], np.uint8)
TRI_RGB = np.array([58, 112, 61], np.uint8)

im = np.array(Image.open(SRC).convert('RGB'))
box = np.all(im == BOX_RGB, axis=2)
ys, xs = np.nonzero(box)
x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
# box sprite (mask) and its top-left origin
sprite = box[y0:y1 + 1, x0:x1 + 1]
bh, bw = sprite.shape

# background = frame with box removed (box sits on pure white)
bg = im.copy()
bg[box] = 255

# destination: same-size box centered on the right object (triangle)
tri = np.all(im == TRI_RGB, axis=2)
ty, tx = np.nonzero(tri)
cx, cy = (tx.min() + tx.max()) / 2, (ty.min() + ty.max()) / 2
dx0, dy0 = int(round(cx - bw / 2 + 0.5)), int(round(cy - bh / 2 + 0.5))

def ease(t):  # smoothstep
    return t * t * (3 - 2 * t)

os.makedirs('/app/output', exist_ok=True)
frames = []
for i in range(N):
    t = ease(i / (N - 1))
    px = int(round(x0 + (dx0 - x0) * t))
    py = int(round(y0 + (dy0 - y0) * t))
    f = bg.copy()
    region = f[py:py + bh, px:px + bw]
    region[sprite] = BOX_RGB
    frames.append(f)

# frame 0 must equal the source exactly
assert np.array_equal(frames[0], im)

p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                      '-s', f'{im.shape[1]}x{im.shape[0]}', '-r', str(FPS), '-i', '-',
                      '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '12', '-preset', 'slow', OUT],
                     stdin=subprocess.PIPE)
for f in frames:
    p.stdin.write(f.tobytes())
p.stdin.close(); p.wait()
print('wrote', OUT, 'dest box top-left', dx0, dy0)
