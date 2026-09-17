from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import subprocess

ROOT=Path('/app')

def main():
    src=np.array(Image.open(ROOT/'first_frame.png').convert('RGB'))
    background=src.copy()
    objects=[]
    for x0,y0,x1,y1,dx in [(175,477,341,643,478),(384,381,482,480,304)]:
        patch=src[y0:y1,x0:x1].copy()
        mask=np.any(patch!=255,axis=2)
        layer=np.zeros((1024,1024,4),np.uint8)
        layer[y0:y1,x0:x1,:3]=patch
        layer[y0:y1,x0:x1,3]=mask.astype(np.uint8)*255
        background[y0:y1,x0:x1]=255
        objects.append((layer,dx))
    out=ROOT/'output'
    out.mkdir(exist_ok=True)
    p=subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    for i in range(30):
        if i==0:
            frame=src
        else:
            t=i/29
            progress=t*t*(3-2*t)
            frame=background.astype(np.float32)
            for layer,dx in objects:
                moved=cv2.warpAffine(layer,np.float32([[1,0,dx*progress],[0,1,0]]),(1024,1024),flags=cv2.INTER_NEAREST,borderValue=0)
                m=moved[:,:,3]>0
                frame[m]=moved[:,:,:3][m]
            frame=frame.astype(np.uint8)
        p.stdin.write(frame.tobytes())
    p.stdin.close()
    err=p.stderr.read()
    if p.wait():
        raise RuntimeError(err.decode())

if __name__=='__main__':
    main()
