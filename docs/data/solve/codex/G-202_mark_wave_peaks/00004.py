from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = Image.open(ROOT / 'first_frame.png').convert('RGB')
a = np.asarray(base)
mask = a.min(axis=2) < 100
# Image y increases downward, so the wave maxima are minima of its y trace.
xs = np.where(mask.any(axis=0))[0]
y = np.array([np.where(mask[:, x])[0].mean() for x in xs])
candidates = []
for i in range(15, len(xs)-15):
    if y[i] == y[i-15:i+16].min():
        if not candidates or xs[i] - candidates[-1][0] > 50:
            candidates.append((int(xs[i]), float(y[i])))
        elif y[i] < candidates[-1][1]:
            candidates[-1] = (int(xs[i]), float(y[i]))
# Center each flat rasterized apex on its full plateau.
peaks = []
for x, v in candidates:
    ids = (abs(xs-x) < 15) & (y <= v + 0.5)
    peaks.append((float(xs[ids].mean()), float(y[ids].min())))

frames = []
for frame in range(10):
    im = base.copy()
    # Each marker takes three frames to trace, with earlier markers retained.
    for j, (x, yy) in enumerate(peaks):
        progress = min(1., max(0., (frame - j*3)/3))
        if progress == 0:
            continue
        scale = 4
        layer = Image.new('RGBA', (1024*scale, 1024*scale))
        draw = ImageDraw.Draw(layer)
        r = 23
        box = tuple(int(z*scale) for z in (x-r, yy-r, x+r, yy+r))
        red = (235, 20, 30, 255)
        draw.arc(box, -90, -90+360*progress, fill=red, width=3*scale)
        d = 4
        draw.ellipse(tuple(int(z*scale) for z in (x-d, yy-d, x+d, yy+d)), fill=red)
        layer = layer.resize(base.size, Image.Resampling.LANCZOS)
        im.paste(layer, (0, 0), layer)
    frames.append(im)

proc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
    '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16',
    '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '0',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUT / 'video.mp4')], stdin=subprocess.PIPE)
for im in frames:
    proc.stdin.write(im.tobytes())
proc.stdin.close()
if proc.wait():
    raise RuntimeError('ffmpeg failed')
print('Peaks:', peaks)
