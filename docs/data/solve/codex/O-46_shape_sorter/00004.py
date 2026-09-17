from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
background = np.array([248, 250, 252], dtype=np.uint8)
colors = [(96, 165, 250), (248, 113, 113), (34, 211, 238)]
# Translate the original silhouettes, retaining every outline and other scene pixel.
deltas = [(566, -56), (422, 162), (563, 57)]
shapes = []
base = original.copy()
for color in colors:
    mask = np.all(original == color, axis=2)
    y, x = np.nonzero(mask)
    shapes.append((y, x, color))
    base[mask] = background

encoder = subprocess.Popen([
    'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
    '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
    '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow',
    '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    str(OUT / 'video.mp4')
], stdin=subprocess.PIPE)
for frame in range(78):
    canvas = base.copy()
    for i, ((y, x, color), (dx, dy)) in enumerate(zip(shapes, deltas)):
        t = np.clip((frame - (2 + i * 24)) / 23.0, 0.0, 1.0)
        t = t * t * (3.0 - 2.0 * t)
        canvas[y + round(dy * t), x + round(dx * t)] = color
    if frame == 0:
        assert np.array_equal(canvas, original)
    encoder.stdin.write(canvas.tobytes())
encoder.stdin.close()
if encoder.wait() != 0:
    raise RuntimeError('Video encoding failed')
