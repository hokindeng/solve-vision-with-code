from pathlib import Path
import math
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = Image.open(ROOT / 'first_frame.png').convert('RGB')
# The blue and green centers lie on a 45-degree line. Their shared
# boundary is 102.5 pixels from the blue center toward the green one.
cx = 465 + 102.5 / math.sqrt(2)
cy = 226 + 102.5 / math.sqrt(2)
radius = 22
scale = 4
encoder = subprocess.Popen([
    'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
    '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
    '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '15',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUT / 'video.mp4')
], stdin=subprocess.PIPE)
for frame in range(60):
    image = base.copy()
    if frame > 0:
        progress = min(frame / 55, 1.0)
        mask = Image.new('L', (1024 * scale, 1024 * scale), 0)
        draw = ImageDraw.Draw(mask)
        points = []
        for theta in np.linspace(-math.pi / 2, -math.pi / 2 + progress * math.tau,
                                 max(2, int(360 * progress))):
            points.append(((cx + radius * math.cos(theta)) * scale,
                           (cy + radius * math.sin(theta)) * scale))
        draw.line(points, fill=255, width=4 * scale, joint='curve')
        for x, y in (points[0], points[-1]):
            draw.ellipse((x-2*scale, y-2*scale, x+2*scale, y+2*scale), fill=255)
        mask = mask.resize(base.size, Image.Resampling.LANCZOS)
        image.paste((0, 0, 0), (0, 0), mask)
    encoder.stdin.write(image.tobytes())
encoder.stdin.close()
if encoder.wait() != 0:
    raise RuntimeError('Video encoding failed')
