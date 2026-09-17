from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path('/app')

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    encoder = subprocess.Popen([
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
        '-an', '-c:v', 'libx264', '-crf', '18', '-preset', 'slow',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')
    ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    scale = 4
    for index in range(16):
        frame = base.copy()
        if index:
            overlay = Image.new('RGBA', (1024*scale, 1024*scale))
            draw = ImageDraw.Draw(overlay)
            # Enclose the lone irregular polygon without covering its boundary.
            cx, cy, radius = 396, 843, 103
            sweep = 2 * math.pi * index / 15
            points = [((cx + radius * math.cos(-math.pi/2 + sweep*j/360))*scale,
                       (cy + radius * math.sin(-math.pi/2 + sweep*j/360))*scale)
                      for j in range(361)]
            draw.line(points, fill=(235, 25, 35, 255), width=5*scale, joint='curve')
            for x, y in (points[0], points[-1]):
                r = 2.5*scale
                draw.ellipse((x-r, y-r, x+r, y+r), fill=(235,25,35,255))
            overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)
            frame.paste(overlay, (0,0), overlay)
        encoder.stdin.write(np.asarray(frame).tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
