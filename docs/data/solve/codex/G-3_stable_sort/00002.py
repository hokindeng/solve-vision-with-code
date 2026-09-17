from pathlib import Path
import cv2
import numpy as np
import subprocess

ROOT=Path('/app')
OUT=ROOT/'output'
OUT.mkdir(exist_ok=True)
im=cv2.imread(str(ROOT/'first_frame.png'))
bg=im[0,0].copy()
n,labels,stats,_=cv2.connectedComponentsWithStats(np.any(im!=bg,axis=2).astype(np.uint8),8)
# Preserve each original raster, including its one-pixel outline.
shapes={}
for k in range(1,n):
    x,y,w,h,area=stats[k]
    shapes[k]=(im[y:y+h,x:x+w].copy(),labels[y:y+h,x:x+w]==k,np.array([x+(w-1)/2,y+(h-1)/2],float))
order=[1,6,4,3,2,5] # Circles, then triangles, ascending in size.
width=sum(stats[k,2] for k in order)
gap=24
left=(1024-width-gap*5)//2
goals={}
for k in order:
    w=stats[k,2]
    goals[k]=np.array([left+(w-1)/2,512.0])
    left+=w+gap
# Upper arcs let the two small shapes pass above the larger shapes.
controls={1:([600,110],[84,110]),3:([183,225],[525,225])}
proc=subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','bgr24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','15','-preset','slow','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')],stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
for frame in range(96):
    t=np.clip((frame-3)/88,0,1)
    u=t*t*(3-2*t)
    canvas=np.empty_like(im);canvas[:]=bg
    for k in order:
        patch,mask,start=shapes[k]
        end=goals[k]
        if k in controls:
            a,b=map(np.array,controls[k])
            pos=(1-u)**3*start+3*(1-u)**2*u*a+3*(1-u)*u*u*b+u**3*end
        else:
            pos=start+(end-start)*u
        h,w=mask.shape
        x,y=np.rint(pos-np.array([(w-1)/2,(h-1)/2])).astype(int)
        canvas[y:y+h,x:x+w][mask]=patch[mask]
    if frame==0:
        assert np.array_equal(canvas,im)
    proc.stdin.write(canvas.tobytes())
proc.stdin.close()
if proc.wait()!=0: raise RuntimeError('Video encoding failed')
