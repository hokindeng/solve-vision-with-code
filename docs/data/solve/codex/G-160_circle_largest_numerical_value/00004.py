from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')
OUT = ROOT / 'output'

def main():
    OUT.mkdir(exist_ok=True)
    source = Image.open(ROOT / 'first_frame.png').convert('RGB')
    w, h = source.size
    # Values are 59, 85, 62, and 88; only 88 receives a circle.
    # Hold the unchanged scene for comparison, then trace the circle continuously.
    scale = 4
    encoder = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow',
        '-crf', '12', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(OUT / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for frame in range(80):
        image = source.copy()
        progress = min(1.0, max(0.0, (frame - 15) / 57))
        if progress > 0:
            mask = Image.new('L', (w * scale, h * scale), 0)
            draw = ImageDraw.Draw(mask)
            # A true circular stroke comfortably encloses both digits of 88.
            cx, cy, radius = 695, 819, 110
            angles = np.linspace(-np.pi / 2, -np.pi / 2 + 2 * np.pi * progress,
                                 max(2, int(720 * progress)))
            points = [((cx + radius * np.cos(a)) * scale,
                       (cy + radius * np.sin(a)) * scale) for a in angles]
            stroke = 7 * scale
            draw.line(points, fill=255, width=stroke, joint='curve')
            for x, y in (points[0], points[-1]):
                r = stroke / 2
                draw.ellipse((x-r, y-r, x+r, y+r), fill=255)
            mask = mask.resize((w, h), Image.Resampling.LANCZOS)
            image.paste((230, 0, 0), (0, 0), mask)
        encoder.stdin.write(image.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
