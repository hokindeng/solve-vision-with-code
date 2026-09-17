from PIL import Image
import numpy as np
import cv2
import subprocess
from pathlib import Path

ROOT=Path('/app')
base=np.array(Image.open(ROOT/'first_frame.png').convert('RGB'))
color=np.array([229,229,45],dtype=float)
# Reuse the exact silhouette of the given plus, retaining its 161-pixel dimensions.
mask=np.all(base[602:763,108:269]==color,axis=2).astype(np.uint8)
inner=cv2.erode(mask,np.ones((5,5),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
edge=mask-inner

def ease(t):
    t=np.clip(t,0.,1.)
    return t*t*(3-2*t)

def clear_question(frame,cx,amount):
    r=frame[650:713,cx-24:cx+25]
    r[:]=np.round(r*(1-amount)+255*amount).astype(np.uint8)

def plus(frame,cx,cy,opacity=1.,fill=0.):
    x,y=int(round(cx-80)),int(round(cy-80))
    alpha=(edge+inner*fill)*opacity
    r=frame[y:y+161,x:x+161]
    r[:]=np.round(r*(1-alpha[:,:,None])+color*alpha[:,:,None]).astype(np.uint8)

out=ROOT/'output'
out.mkdir(exist_ok=True)
p=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','16','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')],stdin=subprocess.PIPE)
for i in range(64):
    frame=base.copy()
    # First resolve the middle answer, then hollow its interior.
    reveal=ease((i-3)/10)
    hollow=ease((i-13)/17)
    if reveal:
        clear_question(frame,476,reveal)
        plus(frame,476,682,reveal,1-hollow)
    # The right answer repeats the outlined plus, translated down by the
    # same 100 pixels demonstrated by the rectangles above.
    appear=ease((i-32)/8)
    drop=ease((i-40)/20)
    if appear:
        clear_question(frame,764,appear)
        plus(frame,764,682+100*drop,appear,0)
    p.stdin.write(frame.tobytes())
p.stdin.close()
if p.wait()!=0:
    raise RuntimeError('ffmpeg failed')
