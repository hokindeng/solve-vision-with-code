from pathlib import Path
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path(__file__).resolve().parent

def main():
    original = Image.open(ROOT / 'first_frame.png').convert('RGB')
    pixels = np.array(original)
    ink = (pixels.min(axis=2) < 100).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(ink, 8)
    hollow = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if w < 50 or h < 50:
            continue
        cx, cy = x + (w - 1) / 2, y + (h - 1) / 2
        if ink[round(cy), round(cx)] == 0:
            hollow.append((cx, cy, max(w, h) / 2 + 10))
    hollow.sort(key=lambda p: (p[1], p[0]))
    assert len(hollow) == 4, hollow
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
               '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '12',
               '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
               str(out / 'video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    scale = 3
    for frame in range(80):
        mask = Image.new('L', (1024 * scale, 1024 * scale), 0)
        draw = ImageDraw.Draw(mask)
        for j, (cx, cy, radius) in enumerate(hollow):
            progress = min(1.0, max(0.0, (frame - (5 + 18 * j)) / 16))
            if progress <= 0:
                continue
            box = tuple(v * scale for v in (cx-radius, cy-radius, cx+radius, cy+radius))
            draw.arc(box, -90, -90 + 360 * progress, fill=255, width=4*scale)
        mask = mask.resize(original.size, Image.Resampling.LANCZOS)
        image = Image.composite(Image.new('RGB', original.size, (235, 25, 35)), original, mask)
        process.stdin.write(np.asarray(image).tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
