from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = Image.open(ROOT / 'first_frame.png').convert('RGB')
rgb = np.array(base)
# A point is strictly inside the overlap when its entire surrounding
# one-pixel ring has the intersection's color.
overlap = np.all(rgb == (148, 148, 188), axis=2)
black = np.uint8(np.all(rgb < 50, axis=2))
count, labels, stats, centers = cv2.connectedComponentsWithStats(black)
points = []
for label in range(1, count):
    component = np.uint8(labels == label)
    ring = (cv2.dilate(component, np.ones((3, 3), np.uint8)) != 0) & (component == 0)
    if np.all(overlap[ring]):
        points.append(tuple(centers[label]))
points.sort(key=lambda p: (p[1], p[0]))
assert len(points) == 2, points

command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
           '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
           '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '15',
           '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
           str(OUT / 'video.mp4')]
proc = subprocess.Popen(command, stdin=subprocess.PIPE)
scale = 4
for frame in range(37):
    layer = Image.new('RGBA', (1024*scale, 1024*scale))
    draw = ImageDraw.Draw(layer)
    for index, (x, y) in enumerate(points):
        start = 3 + index*15
        progress = min(1.0, max(0.0, (frame-start)/13))
        if progress <= 0:
            continue
        radius = 16
        bounds = tuple(int(v*scale) for v in (x-radius, y-radius, x+radius, y+radius))
        draw.arc(bounds, -90, -90+360*progress, fill=(235, 20, 30, 255), width=3*scale)
    layer = layer.resize(base.size, Image.Resampling.LANCZOS)
    result = Image.alpha_composite(base.convert('RGBA'), layer).convert('RGB')
    proc.stdin.write(np.array(result).tobytes())
proc.stdin.close()
if proc.wait() != 0:
    raise RuntimeError('ffmpeg encoding failed')
print('Created', OUT / 'video.mp4', 'with points:', points)
