from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT=Path('/app')

def main():
    original=np.array(Image.open(ROOT/'first_frame.png').convert('RGB'))
    # Preserve the source artwork, lifting only the orange ball from it.
    ball=(original[:,:,0]>160)&(original[:,:,1]<100)&(original[:,:,2]<60)
    background=original.copy()
    background[ball]=255
    hidden=Image.fromarray(background)
    d=ImageDraw.Draw(hidden)
    d.polygon([(437,593),(464,610),(447,638),(420,621)],fill=(138,43,226),outline=(78,0,166),width=1)
    restored=np.array(hidden)
    background[ball]=restored[ball]
    alpha=ball.astype(np.float32)
    foreground=original.astype(np.float32)*alpha[:,:,None]
    # Each waypoint puts the lower edge of the ball on a platform's top edge.
    platforms=np.array([[442,615],[474,575],[508,539],[540,501],[566,459],[585,415],[602,371]],dtype=float)
    contacts=platforms-np.array([0.,55.])
    points=np.vstack(([416.,659.], contacts))
    output=ROOT/'output'; output.mkdir(exist_ok=True)
    proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output/'video.mp4')],stdin=subprocess.PIPE)
    for frame in range(64):
        if frame==0:
            img=original
        else:
            # Seven short arcs, with a soft final landing and a brief rest.
            travel=min(frame/59.,1.)*7
            i=min(int(travel),6)
            u=min(travel-i,1.)
            blend=u*u*(3-2*u)
            pos=points[i]*(1-blend)+points[i+1]*blend
            pos[1]-=(19 if i else 24)*np.sin(np.pi*u)
            mat=np.array([[1,0,pos[0]-416],[0,1,pos[1]-659]],dtype=np.float32)
            a=cv2.warpAffine(alpha,mat,(1024,1024),flags=cv2.INTER_LINEAR)
            fg=cv2.warpAffine(foreground,mat,(1024,1024),flags=cv2.INTER_LINEAR)
            img=np.clip(fg+background*(1-a[:,:,None]),0,255).astype(np.uint8)
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    if proc.wait()!=0: raise RuntimeError('ffmpeg failed')

if __name__=='__main__': main()
