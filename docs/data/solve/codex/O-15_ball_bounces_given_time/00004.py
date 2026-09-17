from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
first = Image.open(ROOT / 'first_frame.png').convert('RGB')
a = np.array(first)
ball_color = (138,43,226)
arrow_color = (255,140,0)
moving = np.all(a == ball_color,axis=2) | np.all(a == arrow_color,axis=2)
a[moving] = 255
background = Image.fromarray(a)
# The interior wall faces lie at x,y=63 and 962. The ball's radius
# offsets the center's collision planes from those faces.
radius = 30.5
lo, hi = 63 + radius, 962 - radius
p = np.array([732.,435.5])
v = np.array([148.,-58.])
v /= np.linalg.norm(v)
segments = []
length = 0.
for bounce in range(6):
    times = np.array([(hi-p[k])/v[k] if v[k]>0 else (lo-p[k])/v[k] for k in range(2)])
    axis = int(np.argmin(times))
    distance = times[axis]
    q = p + distance*v
    segments.append((length,length+distance,p.copy(),q.copy()))
    length += distance
    p = q
    v[axis] *= -1

def position(s):
    for start,end,p,q in segments:
        if s <= end + 1e-8:
            return p + np.clip((s-start)/(end-start),0,1)*(q-p)
    return segments[-1][3]

# Supersample only the moving disc; unchanged background pixels are copied.
def render(p):
    frame = background.copy()
    x,y = p
    left,top = int(np.floor(x-radius-1)),int(np.floor(y-radius-1))
    size = 65
    scale = 4
    patch = Image.new('RGB',(size*scale,size*scale),'white')
    draw = ImageDraw.Draw(patch)
    draw.ellipse(((x-radius-left)*scale,(y-radius-top)*scale,
                  (x+radius-left)*scale,(y+radius-top)*scale),fill=ball_color)
    patch = patch.resize((size,size),Image.Resampling.LANCZOS)
    # Clip the drawing to the interior so the static wall stays exact.
    x0,y0 = max(left,63),max(top,63)
    x1,y1 = min(left+size,962),min(top+size,962)
    frame.paste(patch.crop((x0-left,y0-top,x1-left,y1-top)),(x0,y0))
    return frame

cmd = ['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24',
       '-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264',
       '-preset','medium','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')]
proc = subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
for i in range(80):
    # The initial arrow is a direction cue, cleared when movement begins.
    frame = first if i == 0 else render(position(length*i/79))
    proc.stdin.write(np.asarray(frame).tobytes())
proc.stdin.close()
if proc.wait():
    raise RuntimeError('ffmpeg failed')
print('Created',OUT/'video.mp4')
print('Six collision centers:',[q.tolist() for _,_,_,q in segments])
