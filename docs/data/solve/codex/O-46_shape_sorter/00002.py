from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
original = Image.open(ROOT / 'first_frame.png').convert('RGB')
a = np.array(original)
bg = (248, 250, 252)
# Preserve the exact source silhouettes and translate them onto the matching slots.
specs = [
    ((250, 204, 21), (493, 8)),       # circle
    ((248, 113, 113), (499, -1)),     # triangle
    ((251, 146, 60), (488, -4)),      # star
    ((34, 211, 238), (490, 2)),       # square
    ((244, 114, 182), (491, -4)),     # hexagon
]
base = a.copy()
layers = []
for color, displacement in specs:
    mask = np.all(a == color, axis=2)
    base[mask] = bg
    yy, xx = np.where(mask)
    x0, y0, x1, y1 = xx.min(), yy.min(), xx.max()+1, yy.max()+1
    rgba = np.zeros((y1-y0+4, x1-x0+4, 4), dtype=np.uint8)
    rgba[2:-2,2:-2,:3] = a[y0:y1,x0:x1]
    rgba[2:-2,2:-2,3] = mask[y0:y1,x0:x1]*255
    layers.append((Image.fromarray(rgba), x0-2, y0-2, displacement))
base = Image.fromarray(base)
cmd = ['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo',
       '-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-',
       '-an','-c:v','libx264','-preset','slow','-crf','12','-pix_fmt','yuv420p',
       '-movflags','+faststart',str(OUT/'video.mp4')]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for frame in range(77):
    canvas = base.copy()
    for i, (sprite, x, y, (dx,dy)) in enumerate(layers):
        t = min(1., max(0., (frame-i*15)/15.))
        t = t*t*(3-2*t)
        px, py = round(x+dx*t), round(y+dy*t)
        canvas.paste(sprite, (px,py), sprite)
    if frame == 0:
        assert np.array_equal(np.array(canvas), a)
    proc.stdin.write(canvas.tobytes())
proc.stdin.close()
if proc.wait() != 0:
    raise RuntimeError('ffmpeg failed')
