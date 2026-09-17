import numpy as np, subprocess, os
from PIL import Image

FIRST = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
FPS, N = 16, 44
C1 = np.array([204, 171, 33]); C2 = np.array([136, 88, 158])
MIX = np.round(C1 * C2 / 255.0).astype(np.uint8)  # (109, 59, 20)

base = np.array(Image.open(FIRST).convert('RGB'))
# locate interior of the mixing zone: flood-fill the white region containing the centre
import cv2
white = np.all(base == 255, axis=2).astype(np.uint8)
n, lab = cv2.connectedComponents(white, connectivity=4)
zone = lab == lab[base.shape[0] // 2, base.shape[1] // 2]
ys, xs = np.where(zone)
ix0, ix1, iy0, iy1 = xs.min(), xs.max(), ys.min(), ys.max()
W, H = ix1 - ix0 + 1, iy1 - iy0 + 1

os.makedirs(os.path.dirname(OUT), exist_ok=True)
frames = []
for i in range(N):
    t = i / (N - 1)
    t = t * t * (3 - 2 * t)  # smoothstep pacing
    f = base.copy()
    if i > 0:
        # fill zone from both pigments toward the centre: left half from the left, right half from the right
        half = W / 2.0
        w = int(round(half * t))
        if i == N - 1: w = int(np.ceil(half))
        if w > 0:
            f[iy0:iy1 + 1, ix0:ix0 + w] = MIX
            f[iy0:iy1 + 1, ix1 + 1 - w:ix1 + 1] = MIX
    frames.append(f)
frames[-1][iy0:iy1 + 1, ix0:ix1 + 1] = MIX

p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                      '-s', f'{base.shape[1]}x{base.shape[0]}', '-r', str(FPS), '-i', '-',
                      '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '12', '-preset', 'slow', OUT],
                     stdin=subprocess.PIPE)
for f in frames: p.stdin.write(np.ascontiguousarray(f).tobytes())
p.stdin.close(); p.wait()
print('mixed', MIX, 'interior', (ix0, iy0, ix1, iy1), 'wrote', OUT)
