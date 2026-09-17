from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
original = Image.open(ROOT / 'first_frame.png').convert('RGB')
# Exact source rectangles and the empty slots in the shelf's 32-pixel grid.
books = [(710, 373, 166), (806, 370, 262), (774, 299, 550), (742, 275, 614)]
sprites = [original.crop((x, y, x + 27, 513)) for x, y, _ in books]
settled = original.copy()
frames = [np.asarray(original).copy()]
for index, ((sx, sy, dx), sprite) in enumerate(zip(books, sprites)):
    # Lift the entire book above the tallest spine before moving sideways.
    settled.paste((255, 255, 255), (sx, sy, sx + 27, 513))
    lift = 260
    positions = [(sx, sy - lift // 2), (sx, sy - lift),
                 (round((sx + dx) / 2), sy - lift), (dx, sy - lift),
                 (dx, sy - lift // 2), (dx, sy)]
    if index == 3:
        positions.pop(0)
    for x, y in positions:
        frame = settled.copy()
        frame.paste(sprite, (x, y))
        frames.append(np.asarray(frame).copy())
    settled.paste(sprite, (dx, sy))
assert len(frames) == 24
assert np.array_equal(frames[0], np.asarray(original))
proc = subprocess.Popen([
    'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
    '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
    '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
    '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    str(OUT / 'video.mp4')], stdin=subprocess.PIPE)
for frame in frames:
    proc.stdin.write(frame.tobytes())
proc.stdin.close()
if proc.wait():
    raise RuntimeError('Video encoding failed')
