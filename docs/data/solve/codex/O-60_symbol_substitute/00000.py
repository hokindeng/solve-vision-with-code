from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
# The symbol interior of position 2; its border and label are untouched.
x0, y0, x1, y1 = 308, 465, 403, 560
old = base[y0:y1, x0:x1].copy()
# Reuse the reference symbol's exact raster, centered on the old symbol.
ref = base[20:137, 888:1006]
mask = (ref[:,:,2] > 200) & (ref[:,:,0] < 100) & (ref[:,:,1] < 100)
yy, xx = np.where(mask)
symbol = ref[yy.min():yy.max()+1, xx.min():xx.max()+1].copy()
new = np.full_like(old, 255)
h, w = symbol.shape[:2]
# Both triangles have their bounding-box center at (355, 512).
left, top = 355 - w//2 - x0, 512 - h//2 - y0
new[top:top+h, left:left+w] = symbol

def smooth(t):
    t = np.clip(t, 0, 1)
    return t*t*(3-2*t)

cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
       '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
       '-an', '-c:v', 'libx264', '-crf', '0', '-preset', 'slow',
       '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUT/'video.mp4')]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
for i in range(52):
    frame = base.copy()
    if i <= 25:
        a = smooth(i/25)
        patch = old.astype(float)*(1-a) + 255*a
    else:
        a = smooth((i-26)/25)
        patch = 255*(1-a) + new.astype(float)*a
    frame[y0:y1,x0:x1] = np.rint(patch).astype(np.uint8)
    proc.stdin.write(frame.tobytes())
proc.stdin.close()
err = proc.stderr.read()
if proc.wait():
    raise RuntimeError(err.decode())
