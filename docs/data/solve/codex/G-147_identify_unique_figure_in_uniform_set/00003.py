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
mask = np.uint8(np.any(a != 255, axis=2))
count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
# The squares fill their bounding boxes; the unique circle has lower occupancy.
unique = min(range(1, count), key=lambda i: stats[i, cv2.CC_STAT_AREA] / (stats[i, cv2.CC_STAT_WIDTH] * stats[i, cv2.CC_STAT_HEIGHT]))
cx, cy = centers[unique]
radius = max(stats[unique, cv2.CC_STAT_WIDTH], stats[unique, cv2.CC_STAT_HEIGHT]) / 2 + 13
scale = 4
box = tuple(round(v * scale) for v in (cx-radius, cy-radius, cx+radius, cy+radius))
cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '15', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUT / 'video.mp4')]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for frame in range(60):
    result = base.copy()
    # Pause for inspection, trace the answer clockwise, then hold the solution.
    progress = min(1.0, max(0.0, (frame - 11) / 38))
    if progress > 0:
        overlay = Image.new('RGBA', (1024*scale, 1024*scale))
        draw = ImageDraw.Draw(overlay)
        draw.arc(box, -90, -90 + 360*progress, fill=(235, 30, 40, 255), width=5*scale)
        overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)
        result.paste(overlay, (0, 0), overlay)
    proc.stdin.write(np.asarray(result).tobytes())
proc.stdin.close()
if proc.wait() != 0:
    raise RuntimeError('ffmpeg failed')
