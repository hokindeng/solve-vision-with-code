from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path('/app')

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    pixels = np.asarray(base)
    mask = np.any(pixels < 245, axis=2).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    rectangles = [tuple(map(int, s[:4])) for s in stats[1:] if s[4] > 100]
    # Compare the width-to-height ratios before choosing one rectangle.
    for x, y, w, h in rectangles:
        print(f'Rectangle ({x}, {y}): {w} x {h}; ratio {w/h:.4f}; distance from 1 = {abs(w/h-1):.4f}')
    x, y, w, h = min(rectangles, key=lambda r: abs(r[2]/r[3]-1))
    cx, cy = x + (w-1)/2, y + (h-1)/2
    radius = np.hypot(w/2, h/2) + 13
    scale = 4
    bounds = tuple(v*scale for v in (cx-radius, cy-radius, cx+radius, cy+radius))
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
               '-an', '-c:v', 'libx264', '-crf', '15', '-preset', 'medium',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out/'video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    # Pause to inspect the four ratios, trace one circle, then hold the answer.
    for frame in range(48):
        result = base.copy()
        if frame >= 12:
            progress = min(1.0, (frame-11)/30)
            overlay = Image.new('RGBA', (1024*scale, 1024*scale))
            draw = ImageDraw.Draw(overlay)
            draw.arc(bounds, start=-90, end=-90+360*progress,
                     fill=(235, 25, 35, 255), width=5*scale)
            overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)
            result.paste(overlay, (0, 0), overlay)
        process.stdin.write(np.asarray(result).tobytes())
    process.stdin.close()
    stderr = process.stderr.read()
    if process.wait() != 0:
        raise RuntimeError(stderr.decode())

if __name__ == '__main__':
    main()
