from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = Image.open(ROOT / 'first_frame.png').convert('RGB')
a = np.asarray(base)
h, w = a.shape[:2]
# The connected central color region is the innermost concentric square.
color = a[h // 2, w // 2]
ys, xs = np.where(np.all(a == color, axis=2))
x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
# Put the stroke inside the square's boundary.
inset = 3
corners = [(x0+inset,y0+inset),(x1-inset,y0+inset),
           (x1-inset,y1-inset),(x0+inset,y1-inset),(x0+inset,y0+inset)]
cmd = ['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo',
       '-pix_fmt','rgb24','-s',f'{w}x{h}','-r','16','-i','-',
       '-an','-c:v','libx264','-crf','0','-preset','medium',
       '-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for frame in range(85):
    im = base.copy()
    draw = ImageDraw.Draw(im)
    # Brief observation, then four consecutive side-drawing steps, then hold.
    progress = max(0.0, min(4.0, (frame - 5) / 18.0))
    for side in range(4):
        portion = min(1.0, max(0.0, progress-side))
        if portion == 0:
            continue
        start, end = corners[side], corners[side+1]
        tip = tuple(round(start[k] + portion*(end[k]-start[k])) for k in range(2))
        draw.line([start,tip], fill=(0,0,255), width=7)
    proc.stdin.write(im.tobytes())
proc.stdin.close()
if proc.wait() != 0:
    raise RuntimeError('ffmpeg failed')
