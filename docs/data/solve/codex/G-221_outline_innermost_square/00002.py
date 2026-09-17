from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = Image.open(ROOT / 'first_frame.png').convert('RGB')
a = np.asarray(base)
# The center's color identifies the smallest concentric square.
center_color = a[a.shape[0] // 2, a.shape[1] // 2]
ys, xs = np.where(np.all(a == center_color, axis=2))
x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
width = 6
# Inset the stroke so the outline modifies only the innermost square.
inset = width // 2
corners = [(x0+inset,y0+inset),(x1-inset,y0+inset),
           (x1-inset,y1-inset),(x0+inset,y1-inset),(x0+inset,y0+inset)]
proc = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo',
    '-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-',
    '-an','-c:v','libx264','-crf','0','-preset','medium','-pix_fmt','yuv420p',
    '-movflags','+faststart',str(OUT/'video.mp4')], stdin=subprocess.PIPE)
for frame in range(85):
    im = base.copy()
    draw = ImageDraw.Draw(im)
    # A brief initial view, then trace the four sides in sequence.
    progress = min(4.0, max(0.0, (frame-8)/17.0))
    for edge in range(4):
        amount = min(1.0, max(0.0, progress-edge))
        if amount > 0:
            start, end = corners[edge], corners[edge+1]
            stop = tuple(round(start[j]+amount*(end[j]-start[j])) for j in range(2))
            draw.line([start, stop], fill=(0,0,255), width=width)
    proc.stdin.write(im.tobytes())
proc.stdin.close()
if proc.wait() != 0:
    raise RuntimeError('ffmpeg failed')
