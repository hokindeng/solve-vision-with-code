from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import subprocess

ROOT=Path('/app')
OUT=ROOT/'output/video.mp4'

def ease(t):
    t=np.clip(t,0,1)
    return t*t*(3-2*t)

def main():
    OUT.parent.mkdir(exist_ok=True)
    original=np.array(Image.open(ROOT/'first_frame.png').convert('RGB'))
    colors=[(60,179,113),(70,130,180),(255,215,0),(255,140,0)]
    centers=[(247.,164.),(863.,861.),(319.,759.),(759.,253.)]
    masks=[np.all(original==c,axis=2) for c in colors]
    clean=original.copy()
    clean[np.all(original==0,axis=2)]=255
    radii=[23.5,34.,49.,66.,86.]
    start=np.array([642.,723.])
    proc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT)],stdin=subprocess.PIPE)
    for f in range(108):
        if f<5:
            frame=original.copy()
        else:
            elapsed=f-5
            stage=min(elapsed//24,3)
            local=min(elapsed-stage*24,24)
            src=start if stage==0 else np.array(centers[stage-1])
            dst=np.array(centers[stage])
            # Eighteen frames of travel followed by six frames of growth.
            pos=src+(dst-src)*ease(local/18.)
            radius=radii[stage]+(radii[stage+1]-radii[stage])*ease((local-18)/6.)
            frame=clean.copy()
            for j,mask in enumerate(masks):
                if j<stage or (j==stage and local>=18):
                    frame[mask]=255
            # Fixed-point circle drawing provides smooth subpixel movement.
            cv2.circle(frame,tuple(np.round(pos*256).astype(int)),round(radius*256),(0,0,0),-1,lineType=cv2.LINE_AA,shift=8)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait()!=0:
        raise RuntimeError('ffmpeg failed')

if __name__=='__main__':
    main()
