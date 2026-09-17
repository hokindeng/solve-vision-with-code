from pathlib import Path
import math
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    w, h = base.size
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
               '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
               '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
               str(OUT / 'video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    # The upper-left polygon has unequal corners and no reflection symmetry.
    cx, cy, radius = 141, 283, 77
    scale = 4
    for i in range(16):
        frame = base.copy()
        if i:
            mask = Image.new('L', (w * scale, h * scale), 0)
            draw = ImageDraw.Draw(mask)
            sweep = 2 * math.pi * i / 15
            points = [((cx + radius * math.cos(-math.pi / 2 + sweep * j / 500)) * scale,
                       (cy + radius * math.sin(-math.pi / 2 + sweep * j / 500)) * scale)
                      for j in range(501)]
            draw.line(points, fill=255, width=4 * scale, joint='curve')
            for x, y in (points[0], points[-1]):
                draw.ellipse((x-2*scale, y-2*scale, x+2*scale, y+2*scale), fill=255)
            mask = mask.resize((w, h), Image.Resampling.LANCZOS)
            frame.paste((255, 0, 0), (0, 0), mask)
        proc.stdin.write(np.asarray(frame).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
