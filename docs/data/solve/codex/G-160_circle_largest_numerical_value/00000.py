from pathlib import Path
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)

def main():
    base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    height, width = base.shape[:2]
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
        '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(OUT / 'video.mp4')
    ], stdin=subprocess.PIPE)
    # Pause to compare the six values, then trace just the largest: 97.
    scale = 4
    for frame in range(80):
        progress = min(1.0, max(0.0, (frame - 15) / 58))
        result = base.copy()
        if progress > 0:
            mask = Image.new('L', (width * scale, height * scale), 0)
            draw = ImageDraw.Draw(mask)
            cx, cy, radius = 668, 683, 110
            angles = np.linspace(-math.pi / 2, -math.pi / 2 + 2 * math.pi * progress,
                                 max(2, int(800 * progress)))
            points = [(round((cx + radius * math.cos(a)) * scale),
                       round((cy + radius * math.sin(a)) * scale)) for a in angles]
            stroke = 6 * scale
            draw.line(points, fill=255, width=stroke, joint='curve')
            for x, y in (points[0], points[-1]):
                draw.ellipse((x-stroke/2, y-stroke/2, x+stroke/2, y+stroke/2), fill=255)
            alpha = np.asarray(mask.resize((width, height), Image.Resampling.LANCZOS)) / 255.0
            selected = alpha > 0
            a = alpha[selected, None]
            result[selected] = np.rint(base[selected] * (1-a) + np.array([230, 0, 0]) * a).astype(np.uint8)
        proc.stdin.write(result.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
