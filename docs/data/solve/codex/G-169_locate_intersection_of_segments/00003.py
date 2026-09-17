from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = Image.open(ROOT / 'first_frame.png').convert('RGB')
a = np.asarray(base)
y, x = np.where((a[:,:,2] > 180) & (a[:,:,0] < 160) & (a[:,:,1] < 160))
# Fit each stroke away from the crossing, where they are unambiguous.
sel1 = y < 420
sel2 = (x < 690) & (y > 490)
m1, b1 = np.polyfit(x[sel1], y[sel1], 1)
m2, b2 = np.polyfit(x[sel2], y[sel2], 1)
cx = (b2-b1)/(m1-m2)
cy = m1*cx+b1
radius = 30
scale = 4
box = ((cx-radius)*scale, (cy-radius)*scale,
       (cx+radius)*scale, (cy+radius)*scale)
cmd = ['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24',
       '-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264',
       '-preset','slow','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',
       str(OUT/'video.mp4')]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
for i in range(30):
    frame = base.copy()
    if i >= 3:
        progress = min(1.0, (i-2)/26)
        mask = Image.new('L', (1024*scale,1024*scale))
        draw = ImageDraw.Draw(mask)
        end = -90 + 360*progress
        draw.arc(box, -90, end, fill=255, width=4*scale)
        # Rounded tip on the growing circular stroke.
        if progress < 1:
            angle = np.deg2rad(end)
            r = radius-2
            tx, ty = (cx+r*np.cos(angle))*scale, (cy+r*np.sin(angle))*scale
            draw.ellipse((tx-2*scale,ty-2*scale,tx+2*scale,ty+2*scale),fill=255)
        mask = mask.resize(base.size, Image.Resampling.LANCZOS)
        frame.paste((230,35,40), (0,0), mask)
    p.stdin.write(frame.tobytes())
p.stdin.close()
err = p.stderr.read()
if p.wait():
    raise RuntimeError(err.decode())
print(f'Intersection: ({cx:.2f}, {cy:.2f}); wrote {OUT / "video.mp4"}')
