from pathlib import Path
import math
import cv2
import numpy as np
from PIL import Image
import subprocess

ROOT=Path('/app')

def main():
    original=np.array(Image.open(ROOT/'first_frame.png').convert('RGB'))
    pink=(original[:,:,0]>150)&(original[:,:,1]<150)&(original[:,:,2]>70)
    background=original.copy()
    background[pink]=255
    # Reconstruct the small part of the first identical platform hidden by the ball.
    tile=original[563:583,423:443]
    for y in range(20):
        for x in range(20):
            xx,yy=406+x,577+y
            if pink[yy,xx]:
                background[yy,xx]=tile[y,x]
    ys,xs=np.where(pink)
    center=np.array([(xs.min()+xs.max())/2,(ys.min()+ys.max())/2])
    # The plain, unmarked ball is retained as an exact-color sprite.
    alpha=pink.astype(np.float32)
    foreground=original.astype(np.float32)*alpha[:,:,None]
    first=np.array([415.5,586.5])
    last=np.array([641.5,410.5])
    direction=(last-first)/np.linalg.norm(last-first)
    normal=np.array([direction[1],-direction[0]])
    radius=np.linalg.norm(center-first)
    start_angle=math.atan2(*(center-first)[::-1])
    end_angle=math.atan2(normal[1],normal[0])
    while end_angle<start_angle:
        end_angle+=2*math.pi
    arc_length=radius*(end_angle-start_angle)
    track_length=np.linalg.norm(last-first)
    total=arc_length+track_length
    out=ROOT/'output';out.mkdir(exist_ok=True)
    proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')],stdin=subprocess.PIPE)
    for frame in range(64):
        t=min(frame/59,1.)
        progress=t*t*(3-2*t)
        distance=total*progress
        if distance<arc_length:
            angle=start_angle+distance/radius
            pos=first+radius*np.array([math.cos(angle),math.sin(angle)])
        else:
            u=min((distance-arc_length)/track_length,1.)
            # Small smooth rises between consecutive platform contacts.
            hop=2.0*math.sin(math.pi*(u*13 % 1))**2
            pos=first+u*(last-first)+(radius+hop)*normal
        if frame==0:
            result=original
        else:
            dx,dy=pos-center
            transform=np.float32([[1,0,dx],[0,1,dy]])
            a=cv2.warpAffine(alpha,transform,(1024,1024),flags=cv2.INTER_LINEAR)
            f=cv2.warpAffine(foreground,transform,(1024,1024),flags=cv2.INTER_LINEAR)
            result=np.rint(f+background*(1-a[:,:,None])).clip(0,255).astype(np.uint8)
        proc.stdin.write(result.tobytes())
    proc.stdin.close()
    if proc.wait()!=0:
        raise RuntimeError('Video encoding failed')

if __name__=='__main__':
    main()
