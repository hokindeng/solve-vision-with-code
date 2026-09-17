from pathlib import Path
import subprocess
import math
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')
OUT = ROOT / 'output'

def main():
    OUT.mkdir(exist_ok=True)
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    w, h = base.size
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '12',
        '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(OUT / 'video.mp4')
    ], stdin=subprocess.PIPE)
    # The leftmost object is the square at x=213..297, y=893..977.
    # A circle with a small margin fits without touching any other shape.
    scale = 4
    cx, cy, radius = 255, 935, 66
    for i in range(48):
        frame = base.copy()
        progress = min(i / 45.0, 1.0)
        if progress > 0:
            overlay = Image.new('RGBA', (w*scale, h*scale))
            draw = ImageDraw.Draw(overlay)
            steps = max(2, int(600 * progress))
            points = []
            for j in range(steps + 1):
                theta = -math.pi/2 + 2*math.pi*progress*j/steps
                points.append(((cx + radius*math.cos(theta))*scale,
                               (cy + radius*math.sin(theta))*scale))
            draw.line(points, fill=(255, 0, 0, 255), width=5*scale, joint='curve')
            for x, y in (points[0], points[-1]):
                r = 2.5*scale
                draw.ellipse((x-r, y-r, x+r, y+r), fill=(255, 0, 0, 255))
            overlay = overlay.resize((w,h), Image.Resampling.LANCZOS)
            frame.paste(overlay, (0,0), overlay)
        proc.stdin.write(np.asarray(frame).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
