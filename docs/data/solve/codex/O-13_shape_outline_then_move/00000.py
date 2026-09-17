from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import subprocess

ROOT=Path('/app')
def main():
    out=ROOT/'output'
    out.mkdir(exist_ok=True)
    base=np.array(Image.open(ROOT/'first_frame.png').convert('RGB'))
    # Extract the original star so all copies have precisely the same silhouette.
    crop=base[590:760,100:276]
    mask=((crop[:,:,1]>100)&(crop[:,:,0]<80)&(crop[:,:,2]<140)).astype(np.uint8)
    inner=cv2.erode(mask,np.ones((3,3),np.uint8),iterations=1)
    edge=mask-inner
    green=np.array([27,153,67],dtype=float)
    green=np.median(crop[mask.astype(bool)],axis=0)
    question_masks=[]
    for x in (450,738):
        q=np.zeros(base.shape[:2],dtype=bool)
        region=base[650:715,x:x+55]
        q[650:715,x:x+55]=(region.max(axis=2)-region.min(axis=2)<5)&(region.min(axis=2)<220)
        question_masks.append(q)
    def smooth(t):
        t=np.clip(t,0,1)
        return t*t*(3-2*t)
    def star(frame,x,y,opacity,fill):
        a=(edge+inner*fill)*opacity
        patch=frame[y:y+170,x:x+176]
        patch[:]=patch*(1-a[:,:,None])+green*a[:,:,None]
    proc=subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','15','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')],stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
    for i in range(64):
        f=base.astype(float).copy()
        if i:
            # First answer: reveal a copy, then remove only its interior fill.
            reveal=smooth((i-5)/9)
            f[question_masks[0]]=f[question_masks[0]]*(1-reveal)+255*reveal
            star(f,388,590,reveal,1-smooth((i-15)/15))
            # Second answer retains the outline and translates downward by 60 px,
            # exactly the displacement demonstrated by the upper arrows.
            reveal2=smooth((i-33)/8)
            f[question_masks[1]]=f[question_masks[1]]*(1-reveal2)+255*reveal2
            down=round(60*smooth((i-41)/17))
            star(f,676,590+down,reveal2,0)
        proc.stdin.write(np.uint8(np.clip(np.rint(f),0,255)).tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('ffmpeg failed')
if __name__=='__main__':
    main()
