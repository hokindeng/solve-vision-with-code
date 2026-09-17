from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = Image.open(ROOT / 'first_frame.png').convert('RGB')
a = np.array(base)
# A point is strictly interior when its entire one-pixel exterior ring
# has the overlap color. This excludes points touching either boundary.
overlap = np.all(a == (130, 196, 170), axis=2)
n, labels, stats, centers = cv2.connectedComponentsWithStats(
    (a.max(axis=2) < 40).astype(np.uint8), 8)
points = []
for i in range(1, n):
    if stats[i, cv2.CC_STAT_AREA] < 20:
        continue
    component = (labels == i).astype(np.uint8)
    ring = (cv2.dilate(component, np.ones((3, 3), np.uint8)) > 0) & (component == 0)
    if np.all(overlap[ring]):
        points.append(tuple(centers[i]))
points.sort(key=lambda p: (p[1], p[0]))
assert len(points) == 2, points

command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
           '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
           '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '15',
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUT / 'video.mp4')]
proc = subprocess.Popen(command, stdin=subprocess.PIPE)
for frame in range(37):
    # Supersample just the transparent annotation layer; the base is untouched.
    scale = 4
    layer = Image.new('RGBA', (1024 * scale, 1024 * scale))
    draw = ImageDraw.Draw(layer)
    for index, (x, y) in enumerate(points):
        start = 3 + index * 16
        progress = max(0.0, min(1.0, (frame - start) / 13.0))
        if progress > 0:
            radius = 17
            box = tuple(round(v * scale) for v in
                        (x-radius, y-radius, x+radius, y+radius))
            draw.arc(box, -90, -90 + 360 * progress,
                     fill=(235, 25, 35, 255), width=3 * scale)
    layer = layer.resize(base.size, Image.Resampling.LANCZOS)
    result = Image.alpha_composite(base.convert('RGBA'), layer).convert('RGB')
    proc.stdin.write(np.asarray(result).tobytes())
proc.stdin.close()
if proc.wait() != 0:
    raise RuntimeError('ffmpeg encoding failed')
print('Circled points:', points)
