from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_fill_holes
import subprocess

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = Image.open(ROOT / 'first_frame.png').convert('RGB')
a = np.asarray(base)
overlap = binary_fill_holes(np.all(a == (160, 166, 121), axis=2))
count, labels, stats, centers = cv2.connectedComponentsWithStats(np.all(a == 0, axis=2).astype(np.uint8))
points = [tuple(centers[i]) for i in range(1, count) if np.all(overlap[labels == i])]
points.sort(key=lambda p: (p[1], p[0]))
assert len(points) == 2
proc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '15', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUT / 'video.mp4')], stdin=subprocess.PIPE)
for frame in range(37):
    layer = Image.new('RGBA', (4096, 4096))
    draw = ImageDraw.Draw(layer)
    for j, (x, y) in enumerate(points):
        start = 5 + j * 14
        progress = np.clip((frame - start) / 12, 0, 1)
        if progress > 0:
            radius = 15
            box = tuple(round(v * 4) for v in (x-radius, y-radius, x+radius, y+radius))
            draw.arc(box, -90, -90 + 360 * float(progress), fill=(255, 0, 0, 255), width=12)
    layer = layer.resize(base.size, Image.Resampling.LANCZOS)
    result = Image.alpha_composite(base.convert('RGBA'), layer).convert('RGB')
    proc.stdin.write(result.tobytes())
proc.stdin.close()
if proc.wait() != 0:
    raise RuntimeError('ffmpeg encoding failed')
