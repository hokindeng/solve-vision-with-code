from pathlib import Path
import subprocess
import math
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')

def main():
    source = Image.open(ROOT / 'first_frame.png').convert('RGB')
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    process = subprocess.Popen([
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
        '-an', '-c:v', 'libx264', '-crf', '16', '-preset', 'medium',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')
    ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    scale = 4
    for index in range(48):
        frame = source.copy()
        if index > 0:
            progress = min(index / 44.0, 1.0)
            layer = Image.new('RGBA', (1024 * scale, 1024 * scale))
            draw = ImageDraw.Draw(layer)
            points = []
            for step in range(max(2, int(500 * progress)) + 1):
                count = max(2, int(500 * progress))
                angle = -math.pi / 2 + 2 * math.pi * progress * step / count
                points.append(((615 + 83 * math.cos(angle)) * scale,
                               (237 + 83 * math.sin(angle)) * scale))
            draw.line(points, fill=(230, 20, 30, 255), width=5 * scale, joint='curve')
            for x, y in (points[0], points[-1]):
                r = 2.5 * scale
                draw.ellipse((x-r, y-r, x+r, y+r), fill=(230, 20, 30, 255))
            layer = layer.resize(source.size, Image.Resampling.LANCZOS)
            frame.paste(layer, (0, 0), layer)
        process.stdin.write(np.asarray(frame).tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
