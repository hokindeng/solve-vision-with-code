from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
# Reuse the source minus, preserving its outline and exact proportions.
minus = base[665:700,110:251].copy()
minus[np.all(minus == (76,89,153), axis=2)] = (153,153,30)
small = np.array(Image.fromarray(minus).resize((71,18), Image.Resampling.LANCZOS))

def answer_patch(cx, shape):
    x0, x1, y0, y1 = cx-76, cx+77, 650, 715
    target = np.full((y1-y0,x1-x0,3),255,dtype=np.uint8)
    h,w = shape.shape[:2]
    x = cx-w//2-x0
    y = 682-h//2-y0
    target[y:y+h,x:x+w] = shape
    return (x0,x1,y0,y1,target)

patches = [answer_patch(518,minus),answer_patch(854,small)]
def ease(t):
    t = np.clip(t,0,1)
    return t*t*(3-2*t)

cmd = ['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','0','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')]
proc = subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
for i in range(60):
    frame = base.copy()
    for patch,start,end in zip(patches,(7,32),(27,54)):
        x0,x1,y0,y1,target = patch
        alpha = ease((i-start)/(end-start))
        if alpha:
            original = base[y0:y1,x0:x1]
            frame[y0:y1,x0:x1] = np.rint(original*(1-alpha)+target*alpha).astype(np.uint8)
    proc.stdin.write(frame.tobytes())
proc.stdin.close()
if proc.wait():
    raise RuntimeError('ffmpeg failed')
