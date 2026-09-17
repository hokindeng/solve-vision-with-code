from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import cv2
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = Image.open(ROOT / 'first_frame.png').convert('RGB')
pixels = np.asarray(base)
# Compare each colored rectangle, including its one-pixel black boundary.
mask = ((pixels.max(axis=2) != pixels.min(axis=2))).astype(np.uint8)
n, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
rectangles = []
for x, y, w, h, area in stats[1:]:
    if area > 100:
        w, h = int(w) + 2, int(h) + 2
        rectangles.append((abs(w / h - 1), int(x)-1, int(y)-1, w, h))
for error, x, y, w, h in rectangles:
    print(f'Rectangle ({x}, {y}): {w} x {h}; ratio {w/h:.4f}; distance from 1 = {error:.4f}')
_, x, y, w, h = min(rectangles)
cx, cy = x + (w-1)/2, y + (h-1)/2
radius = ((w/2)**2 + (h/2)**2)**0.5 + 12
# Preserve the scene during comparison; then progressively trace one circle.
# Supersample only the circle overlay so the input pixels remain unchanged elsewhere.
scale = 4
proc = subprocess.Popen([
    'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo',
    '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
    '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '0',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUT/'video.mp4')
], stdin=subprocess.PIPE)
for frame in range(48):
    im = base.copy()
    if frame >= 16:
        progress = min(1.0, (frame-15)/26)
        overlay = Image.new('RGBA', (1024*scale, 1024*scale))
        draw = ImageDraw.Draw(overlay)
        angles = np.linspace(-np.pi/2, -np.pi/2 + 2*np.pi*progress, max(2,int(progress*600)))
        points = [((cx+radius*np.cos(a))*scale, (cy+radius*np.sin(a))*scale) for a in angles]
        draw.line(points, fill=(255,0,0,255), width=5*scale, joint='curve')
        overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)
        im = Image.alpha_composite(im.convert('RGBA'), overlay).convert('RGB')
    proc.stdin.write(np.asarray(im).tobytes())
proc.stdin.close()
if proc.wait() != 0:
    raise RuntimeError('ffmpeg failed')
