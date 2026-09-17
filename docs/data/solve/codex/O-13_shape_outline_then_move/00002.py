from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
orange = np.array([229,120,11], dtype=float)
Y = 682.0
R = 80.0
S = 4

def ease(t):
    t = np.clip(t, 0, 1)
    return t*t*(3-2*t)

def disk_mask(x, radius):
    im = Image.new('L', (1024*S, 1024*S), 0)
    ImageDraw.Draw(im).ellipse(((x-radius)*S,(Y-radius)*S,(x+radius)*S,(Y+radius)*S),fill=255)
    return np.asarray(im.resize((1024,1024),Image.Resampling.LANCZOS),dtype=float)/255

masks = {}
def circle(frame, x, fill, opacity=1):
    key = round(float(x),3)
    if key not in masks:
        outer = disk_mask(x,R)
        inner = disk_mask(x,R-1.8)
        masks[key] = (outer,inner)
    outer,inner = masks[key]
    alpha = np.clip(outer-inner+inner*fill,0,1)*opacity
    frame[:] = frame*(1-alpha[:,:,None])+orange*alpha[:,:,None]

def clear_question(frame, x0, x1, strength):
    frame[650:715,x0:x1] = base[650:715,x0:x1]*(1-strength)+255*strength

# Protect the arrows even while the translated answer passes their positions.
arrow_mask = np.zeros((1024,1024),bool)
arrow_mask[674:692,598:643] = True
arrow_mask &= np.any(base < 250,axis=2)
proc = subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo',
    '-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-',
    '-an','-c:v','libx264','-preset','slow','-crf','14','-pix_fmt','yuv420p',
    '-movflags','+faststart',str(OUT/'video.mp4')],stdin=subprocess.PIPE,
    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
for i in range(64):
    frame = base.astype(float).copy()
    if i > 5:
        appear = ease((i-5)/9)
        clear_question(frame,447,501,appear)
        # The first answer changes from solid to the same orange contour.
        fill = 1-ease((i-14)/18)
        circle(frame,477,fill,appear)
    if i >= 35:
        clear_question(frame,738,792,ease((i-35)/7))
        t = ease((i-35)/24)
        # Retain the intermediate answer as a second copy moves to C.
        circle(frame,477+288*t,0,ease((i-35)/4))
    frame[arrow_mask] = base[arrow_mask]
    proc.stdin.write(np.uint8(np.clip(np.rint(frame),0,255)).tobytes())
proc.stdin.close()
if proc.wait():
    raise RuntimeError('ffmpeg failed')
