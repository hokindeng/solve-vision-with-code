from pathlib import Path
import math
import cv2
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
bg = np.array([252,252,252], dtype=np.uint8)
# Original artwork, including its lettering, moves as a rigid domino.
# (left, top, right, bottom, first moving frame)
blocks = [(60,447,114,569,2), (201,447,255,569,8),
          (342,447,396,569,15), (483,356,537,478,23),
          (483,538,537,660,24), (624,337,678,459,32),
          (624,556,678,678,34), (765,319,819,441,42)]
base = original.copy()
sprites = []
for x0,y0,x1,y1,start in blocks:
    left,top,right,bottom=x0-9,y0-2,x1+10,y1+3
    crop=original[top:bottom,left:right].copy()
    mask=np.any(crop != bg, axis=2).astype(np.uint8)*255
    base[top:bottom,left:right]=bg
    sprites.append((crop,mask,left,top,(x0+x1)/2,y1,start))

proc=subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo',
    '-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-',
    '-an','-c:v','libx264','-crf','17','-preset','slow',
    '-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')],stdin=subprocess.PIPE,
    stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
for frame in range(54):
    canvas=base.copy()
    for crop,mask,left,top,px,py,start in sprites:
        u=np.clip((frame-start)/10.0,0,1)
        # Gravity accelerates each fall, followed by a small settling motion.
        if u<0.88:
            angle=78*(u/0.88)**1.65
        else:
            angle=78+1.6*math.sin(math.pi*(u-0.88)/0.12)
        a=math.radians(angle)
        c,s=math.cos(a),math.sin(a)
        mat=np.array([[c,-s,px+c*(left-px)-s*(top-py)],
                      [s,c,py+s*(left-px)+c*(top-py)]],dtype=np.float32)
        rgb=cv2.warpAffine(crop,mat,(1024,1024),flags=cv2.INTER_LINEAR,borderValue=(252,252,252))
        alpha=cv2.warpAffine(mask,mat,(1024,1024),flags=cv2.INTER_LINEAR).astype(np.float32)/255
        # The source background is uniform, so copying the warped crop's full
        # footprint retains the source antialiasing without a second blend.
        footprint=cv2.warpAffine(np.full(mask.shape,255,np.uint8),mat,(1024,1024),flags=cv2.INTER_NEAREST)>0
        active=footprint & (alpha>0)
        canvas[active]=rgb[active]
    if frame==0:
        canvas=original.copy()
    proc.stdin.write(canvas.tobytes())
proc.stdin.close()
error=proc.stderr.read().decode()
if proc.wait():
    raise RuntimeError(error)
print(OUT/'video.mp4')
