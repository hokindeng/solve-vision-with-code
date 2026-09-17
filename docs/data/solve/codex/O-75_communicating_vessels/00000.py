"""Render pressure-driven, viscously damped leveling of communicating oil tubes."""
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
initial = np.array([37., 54., 59.])
equilibrium = initial.mean()
yellow = np.array([255, 227, 75], dtype=float)
columns = [(223, 313), (457, 547), (691, 781)]
fps, count = 16, 53
T = (count - 1) / fps
# A linear hydraulic resistance gives exponential relaxation toward the mean.
# The geometric mobility sets the displayed settling timescale. Subtract the
# tiny terminal residual so the final rendered state is exactly level.
k, gravity, mobility = 2.58, 9.8, 0.35
rate = mobility * gravity / k
terminal = np.exp(-rate * T)
proc = subprocess.Popen([
    'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
    '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
    '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
    '-preset', 'slow', '-crf', '14', '-pix_fmt', 'yuv420p',
    '-movflags', '+faststart', str(OUT / 'video.mp4')
], stdin=subprocess.PIPE)
for i in range(count):
    frame = base.copy()
    if i:
        t = i / fps
        residual = (np.exp(-rate*t) - terminal) / (1-terminal)
        heights = equilibrium + (initial-equilibrium)*residual
        for (left, right), height in zip(columns, heights):
            # The initial frame calibrates zero to row 879 and 10 px/cm.
            surface = 879. - 10.*height
            rows = np.arange(229, 880, dtype=float)
            fill = np.clip(rows + 1 - surface, 0, 1)
            color = 255 + fill[:, None] * (yellow-255)
            frame[229:880, left:right] = np.rint(color[:, None, :]).astype(np.uint8)
    proc.stdin.write(frame.tobytes())
proc.stdin.close()
if proc.wait():
    raise RuntimeError('ffmpeg encoding failed')
