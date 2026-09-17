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
bg = original[0,0].copy()
# The artwork itself supplies each domino's color, outline and lettering.
dominoes = [
    ('START', (42,437,110,534), (76,532), 2),
    ('T1', (169,438,233,533), (201,532), 9),
    ('T2', (294,438,358,533), (326,532), 16),
    ('T3', (419,438,483,533), (451,532), 23),
    ('A1', (544,341,608,436), (576,435), 30),
    ('B1', (544,535,608,630), (576,629), 30),
    ('A2', (669,316,733,411), (701,410), 38),
    ('B2', (669,560,733,655), (701,654), 38),
    ('B3', (794,584,858,679), (826,678), 46),
    ('B4', (919,609,983,704), (951,703), 53),
]
assets = []
for name, box, pivot, start in dominoes:
    x0,y0,x1,y1 = box
    crop = original[y0:y1,x0:x1].copy()
    alpha = (np.any(crop != bg, axis=2)).astype(np.float32)
    assets.append((box,pivot,start,crop,alpha))

proc = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','15','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')],stdin=subprocess.PIPE)
for frame in range(62):
    canvas = original.copy()
    active=[]
    for box,pivot,start,crop,alpha in assets:
        if frame <= start:
            continue
        x0,y0,x1,y1=box
        canvas[y0:y1,x0:x1]=bg
        t=min(1.0,(frame-start)/7.0)
        # Angular acceleration gives the falling slabs weight, then they settle flat.
        angle=math.pi/2*(t*t*(2-t))
        c,s=math.cos(angle),math.sin(angle)
        px,py=pivot
        matrix=np.array([[c,-s,px+c*(x0-px)-s*(y0-py)],
                         [s,c,py+s*(x0-px)+c*(y0-py)]],dtype=np.float32)
        active.append((crop,alpha,matrix))
    for crop,alpha,matrix in active:
        a=cv2.warpAffine(alpha,matrix,(1024,1024),flags=cv2.INTER_LINEAR)
        color=cv2.warpAffine(crop.astype(np.float32)*alpha[:,:,None],matrix,(1024,1024),flags=cv2.INTER_LINEAR)
        canvas=np.clip(color+canvas.astype(np.float32)*(1-a[:,:,None]),0,255).astype(np.uint8)
    proc.stdin.write(canvas.tobytes())
proc.stdin.close()
if proc.wait()!=0:
    raise RuntimeError('Video encoding failed')
