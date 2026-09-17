from pathlib import Path
import math
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')
OUT = ROOT / 'output/video.mp4'

def frame_at(index, original):
    if index == 0:
        return original.copy()
    scale = 4
    mask = Image.new('L', (1024 * scale, 1024 * scale), 0)
    draw = ImageDraw.Draw(mask)
    for cx, cy, radius, start, end in [(384.5, 350.5, 98, 1, 23), (616, 710, 90, 24, 46)]:
        progress = max(0., min(1., (index - start + 1) / (end - start + 1)))
        if progress == 0:
            continue
        points = []
        for t in np.linspace(0, progress * 2 * math.pi, max(2, int(progress * 600))):
            angle = t - math.pi / 2
            points.append(((cx + radius * math.cos(angle)) * scale,
                           (cy + radius * math.sin(angle)) * scale))
        draw.line(points, fill=255, width=3 * scale, joint='curve')
        for x, y in (points[0], points[-1]):
            r = 1.5 * scale
            draw.ellipse((x-r, y-r, x+r, y+r), fill=255)
    mask = mask.resize((1024, 1024), Image.Resampling.LANCZOS)
    result = original.copy()
    result.paste((0, 0, 0), (0, 0), mask)
    return result

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    original = Image.open(ROOT / 'first_frame.png').convert('RGB')
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', '1024x1024', '-r', '16', '-i', '-', '-an', '-c:v', 'libx264',
               '-crf', '15', '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUT)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for index in range(48):
        process.stdin.write(frame_at(index, original).tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
