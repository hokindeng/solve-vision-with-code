from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = Image.open(ROOT / 'first_frame.png').convert('RGB')
a = np.asarray(base)
# The color at the common center identifies the innermost square.
mask = np.all(a == a[a.shape[0] // 2, a.shape[1] // 2], axis=2)
ys, xs = np.where(mask)
x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
# Place the entire six-pixel border inside the square.
w = 6
inset = w // 2
corners = [(x0 + inset, y0 + inset), (x1 - inset + 1, y0 + inset),
           (x1 - inset + 1, y1 - inset + 1), (x0 + inset, y1 - inset + 1)]
corners.append(corners[0])
cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo',
       '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-', '-an',
       '-c:v', 'libx264', '-crf', '0', '-preset', 'medium', '-pix_fmt', 'yuv420p',
       '-movflags', '+faststart', str(OUT / 'video.mp4')]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for frame in range(85):
    im = base.copy()
    draw = ImageDraw.Draw(im)
    # Four consecutive steps trace top, right, bottom and left edges.
    progress = min(4.0, max(0.0, (frame - 4) / 18.0))
    for edge in range(4):
        t = min(1.0, max(0.0, progress - edge))
        if t > 0:
            start, end = corners[edge], corners[edge + 1]
            tip = tuple(round(s + (e - s) * t) for s, e in zip(start, end))
            draw.line([start, tip], fill=(0, 0, 255), width=w)
    p.stdin.write(im.tobytes())
p.stdin.close()
if p.wait() != 0:
    raise RuntimeError('ffmpeg failed')
