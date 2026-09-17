from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = Image.open(ROOT / 'first_frame.png').convert('RGB')
scale = 4
# The sequence alternates small and large yellow squares. Candidate 3 is large.
cx, cy, radius = 626, 854, 83
command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
           '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
           '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '16',
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUT / 'video.mp4')]
proc = subprocess.Popen(command, stdin=subprocess.PIPE)
for index in range(60):
    frame = base.copy()
    progress = min(1.0, max(0.0, (index - 7) / 44))
    if progress > 0:
        overlay = Image.new('RGBA', (1024 * scale, 1024 * scale))
        draw = ImageDraw.Draw(overlay)
        box = tuple(v * scale for v in (cx-radius, cy-radius, cx+radius, cy+radius))
        draw.arc(box, start=-90, end=-90 + 360 * progress,
                 fill=(230, 35, 43, 255), width=6 * scale)
        overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)
        frame.paste(overlay, (0, 0), overlay)
    proc.stdin.write(np.asarray(frame).tobytes())
proc.stdin.close()
if proc.wait() != 0:
    raise RuntimeError('ffmpeg failed')
