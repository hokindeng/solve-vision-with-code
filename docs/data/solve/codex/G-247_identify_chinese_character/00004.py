from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = Image.open(ROOT / 'first_frame.png').convert('RGB')
w, h = base.size
# Render only the annotation at high resolution, then composite over the source.
scale = 4
cx, cy, radius = 460, 364, 94
frames = 48
proc = subprocess.Popen([
    'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
    '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
    '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
    '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    str(OUT / 'video.mp4')
], stdin=subprocess.PIPE)
for i in range(frames):
    frame = base.copy()
    if i:
        progress = i / (frames - 1)
        angles = np.linspace(-np.pi/2, -np.pi/2 + 2*np.pi*progress,
                             max(2, int(720*progress)))
        points = [((cx + radius*np.cos(a))*scale,
                   (cy + radius*np.sin(a))*scale) for a in angles]
        overlay = Image.new('RGBA', (w*scale, h*scale))
        draw = ImageDraw.Draw(overlay)
        width = 5*scale
        draw.line(points, fill=(230, 20, 28, 255), width=width, joint='curve')
        for x, y in (points[0], points[-1]):
            r = width / 2
            draw.ellipse((x-r, y-r, x+r, y+r), fill=(230, 20, 28, 255))
        overlay = overlay.resize((w, h), Image.Resampling.LANCZOS)
        frame.paste(overlay, (0, 0), overlay)
    proc.stdin.write(frame.tobytes())
proc.stdin.close()
if proc.wait() != 0:
    raise RuntimeError('ffmpeg failed')
