from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output' / 'video.mp4'
OUT.parent.mkdir(parents=True, exist_ok=True)
original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
background = original.copy()
# Preserve the original rasterized orange agent, including its dark outline.
x0, y0, x1, y1 = 340, 150, 405, 220
patch = original[y0:y1, x0:x1].copy()
mask = np.any(patch != (50, 200, 50), axis=2)
background[y0:y1, x0:x1][mask] = (50, 200, 50)
sy, sx = np.nonzero(mask)
colors = patch[sy, sx]
# Coordinates are (column, row). Each leg uses exactly Manhattan distance.
start = (3, 1)
targets = [(0, 0), (0, 8), (5, 8), (7, 9), (2, 5), (2, 7)]
position = start
frames = [original]

def render(col, row):
    frame = background.copy()
    dx = int(round(93 * (col - start[0])))
    dy = int(round(93 * (row - start[1])))
    frame[y0 + sy + dy, x0 + sx + dx] = colors
    return frame

for target in targets:
    # Travel vertically first on the blue-to-yellow leg; otherwise horizontally.
    axes = (1, 0) if position == (7, 9) else (0, 1)
    for axis in axes:
        while position[axis] != target[axis]:
            nxt = list(position)
            nxt[axis] += 1 if target[axis] > position[axis] else -1
            nxt = tuple(nxt)
            for step in range(1, 5):
                t = step / 4
                frames.append(render(position[0] + t * (nxt[0] - position[0]),
                                     position[1] + t * (nxt[1] - position[1])))
            position = nxt
    frames.append(render(*position))
while len(frames) < 134:
    frames.append(frames[-1])
assert len(frames) == 134
assert np.array_equal(frames[0], original)
proc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
                         '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16',
                         '-i', '-', '-an', '-c:v', 'libx264', '-crf', '16',
                         '-preset', 'medium', '-pix_fmt', 'yuv420p',
                         '-movflags', '+faststart', str(OUT)], stdin=subprocess.PIPE)
for frame in frames:
    proc.stdin.write(frame.tobytes())
proc.stdin.close()
if proc.wait() != 0:
    raise RuntimeError('ffmpeg encoding failed')
print(OUT)
