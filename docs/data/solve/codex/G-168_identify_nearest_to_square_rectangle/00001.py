from pathlib import Path
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    rgb = np.array(base)
    mask = (np.min(rgb, axis=2) < 240).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    rectangles = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area > 100:
            rectangles.append((abs(w / h - 1), x, y, w, h))
    rectangles.sort()
    for distance, x, y, w, h in rectangles:
        print(f'Rectangle at ({x}, {y}): {w} x {h}, ratio={w/h:.4f}, distance from 1={distance:.4f}')
    _, x, y, w, h = rectangles[0]
    cx, cy = x + (w-1)/2, y + (h-1)/2
    radius = math.hypot(w/2, h/2) + 10
    scale = 4
    command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', '1024x1024', '-r', '16', '-i', '-', '-an', '-c:v', 'libx264',
               '-preset', 'slow', '-crf', '15', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
               str(OUT/'video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    # Allow an initial comparison, then trace the single answer continuously.
    for frame in range(48):
        result = base.copy()
        progress = min(1.0, max(0.0, (frame - 10) / 32))
        if progress > 0:
            overlay = Image.new('RGBA', (1024*scale, 1024*scale))
            draw = ImageDraw.Draw(overlay)
            points = []
            for angle in np.linspace(-math.pi/2, -math.pi/2 + progress*2*math.pi, max(2, int(500*progress))):
                points.append(((cx+radius*math.cos(angle))*scale, (cy+radius*math.sin(angle))*scale))
            draw.line(points, fill=(235, 25, 35, 255), width=4*scale, joint='curve')
            overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)
            result = Image.alpha_composite(result.convert('RGBA'), overlay).convert('RGB')
        proc.stdin.write(np.asarray(result).tobytes())
    proc.stdin.close()
    errors = proc.stderr.read()
    if proc.wait() != 0:
        raise RuntimeError(errors.decode())

if __name__ == '__main__':
    main()
