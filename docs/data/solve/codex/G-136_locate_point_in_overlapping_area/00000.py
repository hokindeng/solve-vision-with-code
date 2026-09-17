from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = Image.open(ROOT / 'first_frame.png').convert('RGB')
a = np.asarray(base)
# The darker green is the intersection of the two translucent shapes.
overlap = np.all(a == (82, 149, 96), axis=2)
black = np.uint8(np.all(a == 0, axis=2))
n, labels, stats, centers = cv2.connectedComponentsWithStats(black, 8)
points = []
for k in range(1, n):
    component = np.uint8(labels == k)
    ring = (cv2.dilate(component, np.ones((3,3), np.uint8)) > 0) & (component == 0)
    # A point is strictly inside only if its entire perimeter is in the overlap.
    if np.all(overlap[ring]):
        points.append(tuple(centers[k]))
points.sort(key=lambda p: (p[1], p[0]))
assert len(points) == 2, points

proc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error',
    '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24',
    '-s', '1024x1024', '-r', '16', '-i', '-', '-an',
    '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
    '-movflags', '+faststart', str(OUT / 'video.mp4')], stdin=subprocess.PIPE)
# Brief initial inspection, then trace each circle in turn, and hold the result.
for frame in range(37):
    image = base.copy()
    for index, (x, y) in enumerate(points):
        start = 5 + index * 13
        progress = min(1., max(0., (frame - start) / 11.))
        if progress == 0:
            continue
        # Antialias a local overlay; all pixels away from the stroke stay exact.
        radius, size, scale = 15, 42, 4
        patch = Image.new('RGBA', (size*scale, size*scale))
        draw = ImageDraw.Draw(patch)
        c = size / 2
        bounds = tuple(int(v*scale) for v in (c-radius,c-radius,c+radius,c+radius))
        draw.arc(bounds, -90, -90+360*progress, fill=(255,0,0,255), width=3*scale)
        patch = patch.resize((size,size), Image.Resampling.LANCZOS)
        image.paste(patch, (round(x-c),round(y-c)), patch)
    proc.stdin.write(np.asarray(image).tobytes())
proc.stdin.close()
if proc.wait() != 0:
    raise RuntimeError('ffmpeg failed')
print('Circled points:', points)
