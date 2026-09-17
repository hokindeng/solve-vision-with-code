from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = Image.open(ROOT / 'first_frame.png').convert('RGB')
a = np.asarray(base)
mask = ((a[:,:,1] > 130) & (a[:,:,0] < 190) & (a[:,:,2] < 100)).astype(np.uint8)
count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
objects = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] > 100]
# The circle occupies about pi/4 of its bounding box; squares fill theirs.
unique = min(objects, key=lambda i: stats[i, cv2.CC_STAT_AREA] / (stats[i, cv2.CC_STAT_WIDTH] * stats[i, cv2.CC_STAT_HEIGHT]))
cx, cy = centers[unique]
radius = max(stats[unique, cv2.CC_STAT_WIDTH], stats[unique, cv2.CC_STAT_HEIGHT]) / 2 + 13
scale = 4
command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUT / 'video.mp4')]
process = subprocess.Popen(command, stdin=subprocess.PIPE)
for frame in range(60):
    result = base.copy()
    # Pause for inspection, trace the answer clockwise, then hold it for review.
    if frame >= 12:
        progress = min(1.0, (frame - 11) / 40)
        overlay = Image.new('RGBA', (1024*scale, 1024*scale))
        draw = ImageDraw.Draw(overlay)
        box = tuple(v*scale for v in (cx-radius, cy-radius, cx+radius, cy+radius))
        draw.arc(box, start=-90, end=-90+360*progress, fill=(235,35,35,255), width=5*scale)
        overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)
        result.paste(overlay, (0,0), overlay)
    process.stdin.write(np.asarray(result).tobytes())
process.stdin.close()
if process.wait():
    raise RuntimeError('ffmpeg failed')
