from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
# Preserve the supplied artwork, removing only the moving orange marker.
mask = np.all(original == [255, 165, 0], axis=2)
ys, xs = np.where(mask)
base = original.copy()
base[mask] = [0, 255, 0]
start = np.array([561., 357.])
# Grid coordinates are (column, row). Each segment has Manhattan-minimal length.
cells = [(5, 3)]
def go(c, r):
    x, y = cells[-1]
    while x != c:
        x += 1 if c > x else -1
        cells.append((x, y))
    while y != r:
        y += 1 if r > y else -1
        cells.append((x, y))
go(0, 3)  # 1
# Descend first, then move right to 2.
cells.extend([(0, 4), (0, 5)])
go(3, 5)  # 2
go(0, 5)  # 3
go(7, 2)  # red end
points = np.array([[51 + 102*c, 51 + 102*r] for c, r in cells], dtype=float)
assert len(points) - 1 == 23
proc = subprocess.Popen([
    'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
    '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
    '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow',
    '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    str(OUT / 'video.mp4')
], stdin=subprocess.PIPE)
for frame in range(89):
    if frame == 0:
        rgb = original
    else:
        # Continuous motion over the duration, with a short completed-state hold.
        progress = min(frame / 84.0, 1.0) * (len(points)-1)
        i = min(int(progress), len(points)-2)
        pos = points[i] + (points[i+1]-points[i]) * (progress-i)
        dx, dy = np.rint(pos-start).astype(int)
        rgb = base.copy()
        rgb[ys+dy, xs+dx] = original[ys, xs]
    proc.stdin.write(rgb.tobytes())
proc.stdin.close()
if proc.wait():
    raise RuntimeError('ffmpeg failed')
