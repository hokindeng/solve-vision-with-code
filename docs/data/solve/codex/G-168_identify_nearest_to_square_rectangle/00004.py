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
    original = Image.open(ROOT / 'first_frame.png').convert('RGB')
    pixels = np.array(original)
    mask = np.any(pixels < 245, axis=2).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    rectangles = []
    for x, y, w, h, area in stats[1:]:
        if area > 100:
            rectangles.append((abs(w / h - 1), int(x), int(y), int(w), int(h)))
    rectangles.sort()
    for score, x, y, w, h in rectangles:
        print(f'Rectangle ({x}, {y}): {w} x {h}; ratio {w/h:.4f}; distance from 1 = {score:.4f}')
    _, x, y, w, h = rectangles[0]
    cx, cy = x + (w - 1) / 2, y + (h - 1) / 2
    radius = math.hypot(w / 2, h / 2) + 12
    # Only the requested circle is overlaid; comparison takes place before drawing.
    scale = 4
    command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
               '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(OUT / 'video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame in range(48):
        result = original.copy()
        if frame >= 8:
            progress = min(1.0, (frame - 7) / 36)
            overlay = Image.new('RGBA', (1024 * scale, 1024 * scale), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            box = tuple(round(v * scale) for v in (cx-radius, cy-radius, cx+radius, cy+radius))
            draw.arc(box, start=-90, end=-90 + 360 * progress, fill=(235, 25, 35, 255), width=5 * scale)
            overlay = overlay.resize(original.size, Image.Resampling.LANCZOS)
            result.paste(overlay, (0, 0), overlay)
        process.stdin.write(np.asarray(result).tobytes())
    process.stdin.close()
    errors = process.stderr.read().decode()
    if process.wait() != 0:
        raise RuntimeError(errors)

if __name__ == '__main__':
    main()
