from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import subprocess

ROOT=Path('/app')
OUT=ROOT/'output'
OUT.mkdir(exist_ok=True)
source=np.array(Image.open(ROOT/'first_frame.png').convert('RGB'))
bg=source[0,0].copy()
# Each original domino is a sprite, including its printed label.
# Coordinates also preserve the START lettering that extends past its sides.
pieces=[
    ((46,449,118,555),(82,553),1,9),
    ((186,449,227,555),(206,553),8,9),
    ((310,449,351,555),(330,553),15,9),
    ((434,449,475,555),(454,553),22,9),
    ((558,356,599,462),(578,460),30,10),
    ((682,335,723,441),(702,439),38,10),
    ((806,314,847,420),(826,418),46,10),
    ((930,293,971,399),(950,397),53,7),
    ((558,542,599,648),(578,646),30,10),
]
sprites=[]
for box,pivot,start,duration in pieces:
    x0,y0,x1,y1=box
    crop=source[y0:y1,x0:x1]
    mask=np.any(crop!=bg,axis=2).astype(np.uint8)*255
    sprites.append((crop.copy(),mask,box,pivot,start,duration))

frames=[]
for frame in range(62):
    canvas=source.copy()
    active=[]
    for crop,mask,box,pivot,start,duration in sprites:
        if frame<=start:
            continue
        x0,y0,x1,y1=box
        region=canvas[y0:y1,x0:x1]
        region[mask>0]=bg
        t=min(1.0,(frame-start)/duration)
        # Gravity-like acceleration followed by a short settling phase.
        angle=(np.pi/2)*(t*t*(2-t))
        c,s=np.cos(angle),np.sin(angle)
        px,py=pivot
        affine=np.array([[c,-s,px+c*(x0-px)-s*(y0-py)],
                         [s,c,py+s*(x0-px)+c*(y0-py)]],dtype=np.float32)
        active.append((crop,mask,affine))
    for crop,mask,affine in active:
        # Premultiplication keeps rotated edges free from dark fringes.
        alpha=mask.astype(np.float32)/255
        color=crop.astype(np.float32)*alpha[:,:,None]
        a=cv2.warpAffine(alpha,affine,(1024,1024),flags=cv2.INTER_LINEAR)
        rgb=cv2.warpAffine(color,affine,(1024,1024),flags=cv2.INTER_LINEAR)
        area=a>0
        canvas[area]=np.clip(rgb[area]+canvas[area]*(1-a[area,None]),0,255).astype(np.uint8)
    frames.append(canvas)

cmd=['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','17','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')]
proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
_,stderr=proc.communicate(b''.join(f.tobytes() for f in frames))
if proc.returncode:
    raise RuntimeError(stderr.decode())
Image.fromarray(frames[-1]).save(OUT/'last_frame.png')
