from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
source = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
# Preserve the original two-color agent, including its outlined edge.
mask = ((source == (255, 200, 0)).all(axis=2) |
        (source == (200, 150, 0)).all(axis=2))
y, x = np.nonzero(mask)
colors = source[y, x].copy()
background = source.copy()
background[y, x] = (0, 100, 255)
# Start (row 8, column 3) to goal (row 2, column 3): six upward
# moves. This unobstructed path meets the Manhattan-distance lower bound.
command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
           '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16',
           '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow',
           '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
           str(OUT / 'video.mp4')]
proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
for frame_number in range(34):
    frame = background.copy()
    offset = round(558 * frame_number / 33)
    frame[y - offset, x] = colors
    if frame_number == 0:
        assert np.array_equal(frame, source)
    proc.stdin.write(frame.tobytes())
proc.stdin.close()
error = proc.stderr.read()
if proc.wait() != 0:
    raise RuntimeError(error.decode())
