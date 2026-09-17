from pathlib import Path
import math
import cv2
import numpy as np
from PIL import Image
import subprocess

ROOT=Path('/app')
S=3

def main():
    original=np.array(Image.open(ROOT/'first_frame.png').convert('RGB'))
    xs=[143,219,278]
    sprites=[original[598:739,x:x+43].copy() for x in xs]
    background=original.copy()
    for x in xs:
        background[598:737,x:x+43]=255
        background[737:739,x:x+43]=[139,90,43]
    out=ROOT/'output'
    out.mkdir(exist_ok=True)
    proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')],stdin=subprocess.PIPE)
    for frame in range(53):
        t=frame/16
        angles=[]
        for start,end in [(.25,2.1),(.95,2.6),(1.55,3.05)]:
            u=np.clip((t-start)/(end-start),0,1)
            angles.append(math.pi/2*(u*u*(3-2*u)))
        for i in [1,0]:
            b=angles[i+1]
            limit=b+math.asin(np.clip(((xs[i+1]-xs[i])*math.cos(b)-42)/140,-1,1))
            angles[i]=min(angles[i],limit)
        canvas=background.copy()
        for i,a in enumerate(angles):
            x=xs[i]
            if a<1e-9:
                canvas[598:739,x:x+43]=sprites[i]
                continue
            c,s=math.cos(a),math.sin(a)
            # Rotate the original numbered face about its bottom right corner.
            matrix=np.array([[c,-s,x+42-c*42+s*140],[s,c,738-s*42-c*140]],dtype=np.float64)*S
            rgba=np.dstack([sprites[i],np.full((141,43),255,dtype=np.uint8)])
            warped=cv2.warpAffine(rgba,matrix,(1024*S,1024*S),flags=cv2.INTER_NEAREST)
            warped=cv2.resize(warped,(1024,1024),interpolation=cv2.INTER_AREA)
            alpha=warped[:,:,3:4].astype(float)/255
            # Unpremultiply the antialiased sprite before compositing.
            rgb=warped[:,:,:3].astype(float)
            canvas=np.clip(rgb+canvas.astype(float)*(1-alpha),0,255).astype(np.uint8)
        if frame==0:canvas=original
        proc.stdin.write(canvas.tobytes())
        if frame==52:Image.fromarray(canvas).save(out/'last_frame.png')
    proc.stdin.close()
    if proc.wait()!=0:raise RuntimeError('ffmpeg failed')

if __name__=='__main__':main()
