from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = np.asarray(Image.open(ROOT / 'first_frame.png').convert('RGB'))
# Locate the largest, orange (41.9%) sector without modifying the source.
mask = np.all(base == (255, 140, 0), axis=2).astype(np.uint8) * 255
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
# Start at the sector's vertex and trace its complete perimeter.
start = np.argmin(np.sum((contour - [512, 543]) ** 2, axis=1))
contour = np.roll(contour, -start, axis=0)
if contour[20, 0] < contour[0, 0]:
    contour = np.concatenate((contour[:1], contour[:0:-1]))
points = np.vstack((contour, contour[0])).astype(float)
lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
distance = np.concatenate(([0.0], np.cumsum(lengths)))
scale = 3
process = subprocess.Popen([
    'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo',
    '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
    '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '0',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUT / 'video.mp4')
], stdin=subprocess.PIPE)
for frame in range(48):
    result = base.copy()
    if frame:
        progress = frame / 47
        end = distance[-1] * progress
        index = min(np.searchsorted(distance, end, side='right') - 1, len(lengths) - 1)
        fraction = (end - distance[index]) / lengths[index]
        tip = points[index] + fraction * (points[index+1] - points[index])
        path = np.vstack((points[:index+1], tip))
        layer = Image.new('L', (1024*scale, 1024*scale), 0)
        draw = ImageDraw.Draw(layer)
        xy = [tuple(p * scale) for p in path]
        draw.line(xy, fill=255, width=6*scale, joint='curve')
        for p in (xy[0], xy[-1]):
            radius = 3*scale
            draw.ellipse((p[0]-radius, p[1]-radius, p[0]+radius, p[1]+radius), fill=255)
        alpha = np.asarray(layer.resize((1024, 1024), Image.Resampling.LANCZOS)).astype(float) / 255
        active = alpha > 0
        result[active] = np.round(base[active]*(1-alpha[active,None]) + np.array([255,0,0])*alpha[active,None]).astype(np.uint8)
    process.stdin.write(result.tobytes())
process.stdin.close()
if process.wait() != 0:
    raise RuntimeError('ffmpeg encoding failed')
