from PIL import Image, ImageDraw
import numpy as np
from scipy.optimize import minimize_scalar
import cv2, subprocess, os

ROOT='/app'
im=np.array(Image.open(ROOT+'/first_frame.png').convert('RGB'))
center=np.array([434.,434.])
poly=np.array([[434.,434.],[493.,436.],[490.,526.],[520.,527.],[518.,587.],[428.,583.]])
y,x=np.where(np.all(im==[100,100,100],axis=2))
points=np.column_stack((x,y))
def rotated(a):
    a=np.deg2rad(a)
    r=np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]])
    return (poly-center)@r.T+center

def err(a):
    v=rotated(a); ds=[]
    for p,q in zip(v,np.roll(v,-1,axis=0)):
        d=q-p
        t=np.clip((points-p)@d/(d@d),0,1)
        ds.append(np.sum((points-p-t[:,None]*d)**2,axis=1))
    return np.mean(np.min(ds,axis=0))
angle=minimize_scalar(err,bounds=(240,255),method='bounded').x
print('Rotation:',angle,flush=True)
# Remove only the original polygon. Preserve all other pixels and the center marker.
base=im.copy()
mask=np.all(im==[50,129,77],axis=2)|np.all(im==[50,50,50],axis=2)
base[mask]=[240,240,240]
# The circular marker is a fixed foreground element.
marker=np.all(im==[0,0,0],axis=2)|np.all(im==[255,255,255],axis=2)
os.makedirs(ROOT+'/output',exist_ok=True)
proc=subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',ROOT+'/output/video.mp4'],stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
for i in range(70):
    if i==0:
        frame=im.copy()
    else:
        t=i/69
        t=t*t*(3-2*t)
        frame=base.copy()
        vertices=np.rint(rotated(angle*t)).astype(np.int32)
        cv2.fillPoly(frame,[vertices],(50,129,77))
        cv2.polylines(frame,[vertices],True,(50,50,50),1,cv2.LINE_8)
        # The polygon's first vertex remains attached to the center throughout.
        frame[marker]=im[marker]
    proc.stdin.write(frame.tobytes())
proc.stdin.close()
assert proc.wait()==0
