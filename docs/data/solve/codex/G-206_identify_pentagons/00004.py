from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output' / 'video.mp4'

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    original = Image.open(ROOT / 'first_frame.png').convert('RGB')
    w, h = original.size
    # The rose-colored polygon is the only pentagon. Its five vertices
    # are approximately (502,295), (598,262), (659,344), (601,425), (503,396).
    cx, cy = 573, 344
    radius = 98
    count = 30
    scale = 4
    encoder = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
        '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(OUT)
    ], stdin=subprocess.PIPE)
    for i in range(count):
        frame = original.copy()
        if i:
            t = i / (count - 1)
            progress = t * t * (3 - 2 * t)
            r = radius * progress
            mask = Image.new('L', (w * scale, h * scale), 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse(((cx-r)*scale, (cy-r)*scale,
                          (cx+r)*scale, (cy+r)*scale),
                         outline=255, width=5*scale)
            mask = mask.resize((w, h), Image.Resampling.LANCZOS)
            frame.paste((255, 0, 0), (0, 0), mask)
        encoder.stdin.write(np.asarray(frame).tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
