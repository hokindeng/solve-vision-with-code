from PIL import Image
import numpy as np
import cv2
import os
import subprocess

ROOT='/app'
def ease(t):
    t=max(0.,min(1.,t))
    return t*t*(3-2*t)

def main():
    src=np.array(Image.open(ROOT+'/first_frame.png').convert('RGB'))
    base=src.copy()
    symbols=[]
    # Only the symbol pixels inside the nine fixed boxes are animated.
    for x in [44,149,254,359,464,569,674,779]:
        crop=src[466:560,x+2:x+96].copy()
        mask=np.any(crop!=255,axis=2)
        yy,xx=np.where(mask)
        symbols.append((yy+466,xx+x+2,crop[yy,xx]))
        base[yy+466,xx+x+2]=255
    ref=src[40:125,900:990]
    mask=(ref[:,:,1]>200)&(ref[:,:,0]<100)&(ref[:,:,2]<100)
    yy,xx=np.where(mask)
    colors=ref[yy,xx]
    # Reference triangle is copied exactly, centered in the first slot.
    tx=xx+900-854
    ty=yy+40+426
    os.makedirs(ROOT+'/output',exist_ok=True)
    tmp=ROOT+'/output/animation.rgb'
    with open(tmp,'wb') as f:
        for i in range(24):
            if i==0:
                frame=src.copy()
            else:
                frame=base.copy()
                shift=round(105*ease(i/10))
                for sy,sx,c in symbols:
                    frame[sy,sx+shift]=c
                alpha=ease((i-10)/5)
                drop=round(-115*(1-ease((i-15)/8)))
                if alpha>0:
                    by=ty+drop
                    frame[by,tx]=np.rint(frame[by,tx]*(1-alpha)+colors*alpha).astype(np.uint8)
            f.write(frame.tobytes())
    subprocess.run(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i',tmp,'-c:v','libx264','-crf','10','-pix_fmt','yuv420p','-movflags','+faststart',ROOT+'/output/video.mp4'],check=True)
    os.remove(tmp)

if __name__=='__main__':
    main()
