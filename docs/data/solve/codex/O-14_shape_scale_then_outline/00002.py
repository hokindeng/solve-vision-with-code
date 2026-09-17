from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = Image.open(ROOT / 'first_frame.png').convert('RGB')
# Match the diagram's green from the source rather than introducing a new color.
a = np.asarray(base)
m = (a[:,:,1] > a[:,:,0] + 20) & (a[:,:,1] > a[:,:,2] + 15)
green = tuple(np.median(a[m], axis=0).astype(int))
S = 4

def smooth(t):
    t = max(0., min(1., t))
    return t*t*(3-2*t)

def pentagon(size, fill_amount):
    im = Image.new('RGB', (128*S, 128*S), 'white')
    d = ImageDraw.Draw(im)
    # Same upward-pointing geometry as the given pentagon.
    points = [(0,-52), (50,-16), (30,42), (-31,42), (-50,-16)]
    pts = [((64+x*size)*S, (64+y*size)*S) for x,y in points]
    inside = tuple(round(255 + fill_amount*(int(c)-255)) for c in green)
    d.polygon(pts, fill=inside)
    d.line(pts+[pts[0]], fill=green, width=5*S, joint='curve')
    return im.resize((128,128), Image.Resampling.LANCZOS)

def stage(frame, cx, progress, size, fill):
    box = (cx-64, 683-64, cx+64, 683+64)
    source = base.crop(box)
    result = pentagon(size, fill)
    frame.paste(Image.blend(source, result, progress), box)

frames = []
for i in range(16):
    frame = base.copy()
    if i:
        # First arrow: reveal a solid pentagon and reduce it to four-fifths size.
        t = min(i/7, 1.)
        stage(frame, 480, smooth(min(t*3,1)), 1-.2*smooth(t), 1.)
        if i >= 8:
            # Second arrow: the smaller solid shape becomes an outline.
            t = (i-7)/8
            stage(frame, 787, smooth(min(t*4,1)), .8, 1-smooth(max(0,(t-.25)/.75)))
    frames.append(np.asarray(frame))

cmd = ['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24',
       '-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264',
       '-preset','slow','-crf','12','-pix_fmt','yuv420p','-movflags','+faststart',
       str(OUT/'video.mp4')]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
_, err = p.communicate(b''.join(f.tobytes() for f in frames))
if p.returncode:
    raise RuntimeError(err.decode())
assert np.array_equal(frames[0], a)
# All drawn changes are confined to the two answer positions.
mask = np.ones(a.shape[:2], dtype=bool)
mask[619:747,416:544] = False
mask[619:747,723:851] = False
assert all(np.array_equal(f[mask],a[mask]) for f in frames)
