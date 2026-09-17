from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT=Path('/app')
base=np.array(Image.open(ROOT/'first_frame.png').convert('RGB'))
# Recover the source silhouette and its precise colour from the supplied diagram.
green=(base[:,:,1]>120)&(base[:,:,0]>100)&(base[:,:,2]<110)
mask=np.zeros((1024,1024),np.uint8)
mask[600:800,80:260]=green[600:800,80:260].astype(np.uint8)*255
contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contour=max(contours,key=cv2.contourArea)
points=cv2.approxPolyDP(contour,1,True).reshape(-1,2).astype(float)
colour=base[700,170].astype(float)
S=4

def silhouette(cx,scale,outline_mix=0):
    pts=(points-np.array([173.,683.]))*scale+np.array([cx,683.])
    im=Image.new('L',(1024*S,1024*S),0)
    d=ImageDraw.Draw(im)
    poly=[tuple(p*S) for p in pts]
    d.polygon(poly,fill=255)
    m=np.array(im.resize((1024,1024),Image.Resampling.LANCZOS))/255.
    if outline_mix:
        # Stroke follows the same polygon; hollowing only removes the interior.
        inner=cv2.erode(np.array(im),np.ones((5*S,5*S),np.uint8))
        inside=np.array(Image.fromarray(inner).resize((1024,1024),Image.Resampling.LANCZOS))/255.
        m=np.clip(m-outline_mix*inside,0,1)
    return m

def ease(t):
    t=np.clip(t,0,1)
    return t*t*(3-2*t)

def place(frame,cx,alpha,scale,hollow=0):
    x=int(cx)
    # Remove only the corresponding placeholder, retaining the rest of the diagram.
    frame[640:725,x-30:x+30]=frame[640:725,x-30:x+30]*(1-alpha)+255*alpha
    m=silhouette(cx,scale,hollow)*alpha
    return frame*(1-m[:,:,None])+colour*m[:,:,None]

out=ROOT/'output'
out.mkdir(exist_ok=True)
proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','12','-pix_fmt','yuv420p',str(out/'video.mp4')],stdin=subprocess.PIPE)
for i in range(16):
    f=base.astype(float).copy()
    if i:
        t=i/7
        f=place(f,480,ease(i/2),1-(1-0.84)*ease(t))
        if i>=8:
            f=place(f,787,ease((i-7)/2),0.84,ease((i-9)/6))
    proc.stdin.write(np.clip(np.rint(f),0,255).astype(np.uint8).tobytes())
proc.stdin.close()
if proc.wait()!=0:
    raise RuntimeError('ffmpeg failed')
