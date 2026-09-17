from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
bg = base[0, 0]
foreground = np.any(base != bg, axis=2).astype(np.uint8)
num, labels, stats, centers = cv2.connectedComponentsWithStats(foreground, 8)
objects = [i for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] > 100]
objects.sort(key=lambda i: (stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_LEFT]))
assert len(objects) == 7
# Keep highlights entirely within each object's original silhouette.
borders = []
for i in objects:
    mask = (labels == i).astype(np.uint8)
    inner = cv2.erode(mask, np.ones((5, 5), np.uint8))
    borders.append((mask - inner).astype(bool))
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 54)
proc = subprocess.Popen([
    'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
    '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
    '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
    '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    str(OUT / 'video.mp4')
], stdin=subprocess.PIPE)
for frame in range(55):
    canvas = base.copy()
    if 1 <= frame <= 49:
        current = (frame - 1) // 7
        canvas[borders[current]] = (255, 155, 0)
    if frame >= 50:
        im = Image.fromarray(canvas)
        draw = ImageDraw.Draw(im)
        draw.text((512, 512), f'Count: {len(objects)}', font=font,
                  fill=(0, 0, 0), anchor='mm')
        canvas = np.asarray(im)
    proc.stdin.write(canvas.tobytes())
proc.stdin.close()
if proc.wait() != 0:
    raise RuntimeError('ffmpeg failed')
