from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    width, height = base.size
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}',
               '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow',
               '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
               str(OUT / 'video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    # The sole pentagon is the blue five-sided shape at bottom right.
    cx, cy, final_radius = 837, 881, 76
    scale = 4
    for index in range(30):
        frame = base.copy()
        if index:
            t = index / 29
            eased = t * t * (3 - 2 * t)
            radius = 2 + (final_radius - 2) * eased
            mask = Image.new('L', (width * scale, height * scale), 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse(tuple(round(v * scale) for v in
                               (cx-radius, cy-radius, cx+radius, cy+radius)),
                         outline=255, width=5 * scale)
            mask = mask.resize(base.size, Image.Resampling.LANCZOS)
            frame.paste((255, 0, 0), (0, 0), mask)
        process.stdin.write(np.asarray(frame).tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
