from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
ball = (original[:,:,0] == 72) & (original[:,:,1] == 61) & (original[:,:,2] == 139)
arrow = (original[:,:,0] == 255) & (original[:,:,1] == 140) & (original[:,:,2] == 0)
background = original.copy()
background[ball | arrow] = 255
# The visible circular outline is centered at (821,819), with radius 30.
# The arrow shaft runs down and left, with direction proportional to (-2,3).
radius = 30.0
start = np.array([821.0,819.0])
v = np.array([-2.0,3.0]); v /= np.linalg.norm(v)
# Walls occupy pixels 53..62 and 962..971, so these are the
# contact limits for the center of a finite-sized ball.
lo, hi = 63.0 + radius, 962.0 - radius
points = [start.copy()]
p = start.copy()
for bounce in range(2):
    times = [(hi-p[k])/v[k] if v[k]>0 else (lo-p[k])/v[k] for k in range(2)]
    axis = int(np.argmin(times))
    p = p + v*times[axis]
    points.append(p.copy())
    v[axis] *= -1
lengths = np.array([np.linalg.norm(points[i+1]-points[i]) for i in range(2)])

def frame(index):
    if index == 0:
        return original
    distance = lengths.sum()*min(index/76.0,1.0)
    segment = 0 if distance <= lengths[0] else 1
    local = distance if segment == 0 else distance-lengths[0]
    c = points[segment] + (points[segment+1]-points[segment])*min(local/lengths[segment],1.0)
    im = Image.fromarray(background)
    d = ImageDraw.Draw(im)
    x,y = c
    d.ellipse((round(x-radius), round(y-radius), round(x+radius), round(y+radius)), fill=(72,61,139))
    return np.asarray(im)

cmd = ['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo',
       '-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-',
       '-an','-c:v','libx264','-preset','slow','-crf','0','-pix_fmt','yuv420p',
       '-movflags','+faststart',str(OUT/'video.mp4')]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for i in range(79):
    proc.stdin.write(frame(i).tobytes())
proc.stdin.close()
if proc.wait() != 0:
    raise RuntimeError('Video encoding failed')
print('Bounce positions:',points[1:])
