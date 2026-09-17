from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = Image.open(ROOT / 'first_frame.png').convert('RGB')
w, h = base.size
frames = 48
scale = 4
command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
           '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', '16', '-i', '-',
           '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
           '-movflags', '+faststart', str(OUT / 'video.mp4')]
proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
for i in range(frames):
    frame = base.copy()
    if i > 0:
        progress = min(i / 44.0, 1.0)
        overlay = Image.new('RGBA', (w * scale, h * scale))
        draw = ImageDraw.Draw(overlay)
        # The lower-right arrow runs counterclockwise, opposite its neighbors.
        cx, cy, radius = 811, 659, 90
        bounds = tuple(v * scale for v in (cx-radius, cy-radius, cx+radius, cy+radius))
        draw.arc(bounds, start=-90, end=-90 + progress * 360,
                 fill=(230, 25, 30, 255), width=4 * scale)
        overlay = overlay.resize((w, h), Image.Resampling.LANCZOS)
        frame.paste(overlay, (0, 0), overlay)
    proc.stdin.write(np.asarray(frame).tobytes())
proc.stdin.close()
if proc.wait() != 0:
    raise RuntimeError('ffmpeg encoding failed')
