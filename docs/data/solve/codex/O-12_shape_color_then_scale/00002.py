from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT=Path('/app')
OUT=ROOT/'output'
OUT.mkdir(exist_ok=True)
base=np.array(Image.open(ROOT/'first_frame.png').convert('RGB'))
sprite=base[592:773,90:271].copy()
purple=np.array([80,80,229]); blue=np.array([53,103,153])
fill=np.all(sprite==purple,axis=2)

def smooth(t):
    t=np.clip(t,0.,1.)
    return t*t*(3-2*t)

def place(frame,cx,opacity,color,scale=1.):
    # Each answer lives in its own original white 181-pixel square.
    x=cx-90; y=592
    patch=sprite.copy()
    patch[fill]=np.rint(purple*(1-color)+blue*color).astype(np.uint8)
    size=round(181*scale)
    if size!=181:
        small=Image.fromarray(patch).resize((size,size),Image.Resampling.LANCZOS)
        patch=np.full((181,181,3),255,dtype=np.uint8)
        off=(181-size)//2
        patch[off:off+size,off:off+size]=np.array(small)
    original=base[y:y+181,x:x+181]
    frame[y:y+181,x:x+181]=np.rint(original*(1-opacity)+patch*opacity).astype(np.uint8)

proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-preset','slow','-crf','15','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')],stdin=subprocess.PIPE)
for i in range(60):
    frame=base.copy()
    if i>5:
        place(frame,518,smooth((i-5)/12),smooth((i-17)/13))
    if i>32:
        place(frame,854,smooth((i-32)/9),1.,1.-(1.-99/127)*smooth((i-41)/14))
    proc.stdin.write(frame.tobytes())
proc.stdin.close()
if proc.wait():
    raise RuntimeError('ffmpeg failed')
