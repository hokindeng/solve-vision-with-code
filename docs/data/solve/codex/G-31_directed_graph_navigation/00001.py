from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
# Extract the original blue triangle, including its dark blue outline.
mask = (original[:, :, 2] > 0) & (original[:, :, 0] == 0) & (original[:, :, 1] == 0)
ys, xs = np.where(mask)
colors = original[ys, xs].copy()
background = original.copy()
background[ys, xs] = (0, 128, 0)
# Directed shortest route: green -> upper white -> red (two edges).
start = np.array([644., 801.])
upper = np.array([627., 287.])
end = np.array([322., 299.])
process = subprocess.Popen([
    'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
    '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
    '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '12',
    '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    str(OUT / 'video.mp4')
], stdin=subprocess.PIPE)
for i in range(30):
    if i <= 14:
        t = i / 14
        position = start + (upper - start) * t
    else:
        t = (i - 14) / 15
        position = upper + (end - upper) * t
    dx, dy = np.rint(position - start).astype(int)
    frame = background.copy()
    frame[ys + dy, xs + dx] = colors
    if i == 0:
        assert np.array_equal(frame, original)
    process.stdin.write(frame.tobytes())
process.stdin.close()
if process.wait() != 0:
    raise RuntimeError('Video encoding failed')
