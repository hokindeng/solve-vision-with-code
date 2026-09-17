from pathlib import Path
import math
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
original = Image.open(ROOT / 'first_frame.png').convert('RGB')
base = original.copy()
# Erase only the moving diagram; this region is plain white behind it.
ImageDraw.Draw(base).rectangle((238,237,462,319), fill='white')
sprite = original.crop((239,238,297,296)).convert('RGBA')
a = np.array(sprite)
a[:,:,3] = np.where(np.all(a[:,:,:3] == 255, axis=2), 0, 255)
sprite = Image.fromarray(a)
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 30*3)
g, restitution = 6.9, .70
# Piecewise exact ballistic arcs. Height measures the ball's clearance.
arcs = []
t = 0.0
h = 15.0
v = -1.8
while True:
    duration = (v + math.sqrt(v*v + 2*g*h))/g
    arcs.append((t, t+duration, h, v))
    t += duration
    impact = v - g*duration
    v = -restitution*impact
    h = 0.0
    # Subpixel rebounds are represented by a stationary contact state.
    if v*v/(2*g)*32 < .35:
        break
stop_time = t

def state(t):
    for start, end, h0, v0 in arcs:
        if t < end:
            dt = t-start
            return max(0., h0+v0*dt-.5*g*dt*dt), v0-g*dt
    return 0., 0.

def frame(i):
    if i == 0:
        return original
    h, v = state(i/16)
    y = 746.5-32*h
    result = base.copy()
    result.paste(sprite, (239, round(y-28.5)), sprite)
    layer = Image.new('RGBA', (3072,3072))
    d = ImageDraw.Draw(layer)
    green = (60,180,60,255)
    def line(points, width):
        d.line([(round(x*3),round(y*3)) for x,y in points], fill=green,width=width*3)
    def poly(points):
        d.polygon([(round(x*3),round(y*3)) for x,y in points],fill=green)
    if abs(v) > .045:
        sign = 1 if v < 0 else -1
        start = y + sign*33
        length = max(5,abs(v)*8.3)
        end = start+sign*length
        head = min(11,length*.65)
        line([(267,start),(267,end-sign*head*.6)],4)
        poly([(267,end),(261,end-sign*head),(273,end-sign*head)])
        text_y = y+18 if sign==1 else y-51
    else:
        text_y = y-16
    d.text((292*3,round(text_y*3)),f'v={abs(v):.1f} m/s',font=font,fill=green)
    layer = layer.resize((1024,1024),Image.Resampling.LANCZOS)
    result.paste(layer,(0,0),layer)
    return result

cmd = ['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo',
       '-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-',
       '-an','-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p',
       '-movflags','+faststart',str(OUT/'video.mp4')]
p = subprocess.Popen(cmd,stdin=subprocess.PIPE)
for i in range(192):
    p.stdin.write(np.asarray(frame(i)).tobytes())
p.stdin.close()
if p.wait():
    raise RuntimeError('ffmpeg failed')
print(f'Created {OUT / "video.mp4"}; ball rests at {stop_time:.3f}s')
