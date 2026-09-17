from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import subprocess

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output'
OUT.mkdir(exist_ok=True)
original=np.array(Image.open(ROOT/'first_frame.png').convert('RGB'))
# The right bottom corner of each domino remains on the original floor.
boxes=[(138,183),(214,259),(269,314),(332,377),(392,437)]
base=original.copy()
sprites=[]
for left,right in boxes:
    sprite=np.zeros((1024,1024,4),np.uint8)
    sprite[583:723,left:right+1,:3]=original[583:723,left:right+1]
    sprite[583:723,left:right+1,3]=255
    sprites.append(sprite)
    base[583:721,left:right+1]=255
    base[721:723,left:right+1]=original[721:723,left-1:left]

def matrix(i,angle):
    return cv2.getRotationMatrix2D((boxes[i][1]+0.5,722.5),-angle,1)

def polygon(i,angle):
    l,r=boxes[i]
    p=np.array([[l-.5,582.5],[r+.5,582.5],[r+.5,722.5],[l-.5,722.5]],np.float32)
    return cv2.transform(p[None],matrix(i,angle))[0]

def cap(i,next_angle):
    other=polygon(i+1,next_angle)
    # Find the first contact with the following solid domino.
    previous=0.
    for angle in np.linspace(0,90,181)[1:]:
        area,_=cv2.intersectConvexConvex(polygon(i,angle),other)
        if area>0.01:
            low,high=previous,float(angle)
            for _ in range(15):
                mid=(low+high)/2
                a,_=cv2.intersectConvexConvex(polygon(i,mid),other)
                if a>0.01: high=mid
                else: low=mid
            return low
        previous=float(angle)
    return 90.

cmd=['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')]
process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
for frame in range(54):
    t=frame/16
    angles=[]
    for i in range(5):
        u=np.clip((t-(0.18+i*0.34))/1.25,0,1)
        angles.append(float(90*u*u*(3-2*u)))
    for i in range(3,-1,-1):
        angles[i]=min(angles[i],cap(i,angles[i+1]))
    if frame==0:
        canvas=original.copy()
    else:
        canvas=base.copy()
        for i in range(4,-1,-1):
            warped=cv2.warpAffine(sprites[i],matrix(i,angles[i]),(1024,1024),flags=cv2.INTER_LINEAR)
            alpha=warped[:,:,3:4].astype(np.float32)/255
            # Warp RGB and alpha separately: recover straight color at edges.
            color=warped[:,:,:3].astype(np.float32)
            canvas=np.clip(color+canvas.astype(np.float32)*(1-alpha),0,255).astype(np.uint8)
    process.stdin.write(canvas.tobytes())
process.stdin.close()
err=process.stderr.read()
if process.wait(): raise RuntimeError(err.decode())
Image.fromarray(canvas).save(OUT/'last_frame.png')
