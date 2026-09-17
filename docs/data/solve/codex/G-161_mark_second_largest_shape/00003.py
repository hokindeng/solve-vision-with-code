from pathlib import Path
import math
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    pixels = np.asarray(base)
    # Locate the blue circle, which is second in descending order of area.
    ys, xs = np.where(np.all(pixels == (65, 105, 225), axis=2))
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    radius = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 + 10
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '15',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for frame in range(40):
        im = base.copy()
        if frame:
            # Smoothly trace a ring clockwise from the top across the duration.
            fraction = min(frame / 38, 1)
            angle = 2 * math.pi * fraction
            count = max(2, int(900 * fraction))
            points = [(cx + radius * math.cos(-math.pi / 2 + angle * j / (count - 1)),
                       cy + radius * math.sin(-math.pi / 2 + angle * j / (count - 1)))
                      for j in range(count)]
            draw = ImageDraw.Draw(im)
            draw.line(points, fill=(255, 0, 0), width=6, joint='curve')
            for x, y in (points[0], points[-1]):
                draw.ellipse((x-2.5, y-2.5, x+2.5, y+2.5), fill=(255, 0, 0))
        proc.stdin.write(im.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
