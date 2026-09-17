from pathlib import Path
import math
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    writer = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '17',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')
    ], stdin=subprocess.PIPE)
    # Hold the unchanged scene for comparison, then trace only the largest value.
    scale = 4
    for frame in range(80):
        progress = min(1.0, max(0.0, (frame - 12) / 59))
        result = base.copy()
        if progress > 0:
            layer = Image.new('RGBA', (1024 * scale, 1024 * scale))
            draw = ImageDraw.Draw(layer)
            count = max(2, int(600 * progress))
            angles = np.linspace(-math.pi / 2, -math.pi / 2 + 2 * math.pi * progress, count)
            points = [((284 + 96 * math.cos(a)) * scale,
                       (413 + 96 * math.sin(a)) * scale) for a in angles]
            color = (230, 20, 30, 255)
            draw.line(points, fill=color, width=5 * scale, joint='curve')
            for x, y in (points[0], points[-1]):
                r = 2.5 * scale
                draw.ellipse((x-r, y-r, x+r, y+r), fill=color)
            layer = layer.resize(base.size, Image.Resampling.LANCZOS)
            result.paste(layer, (0, 0), layer)
        writer.stdin.write(result.tobytes())
    writer.stdin.close()
    if writer.wait() != 0:
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
