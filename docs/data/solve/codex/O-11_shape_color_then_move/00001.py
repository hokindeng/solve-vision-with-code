from PIL import Image
import numpy as np
import subprocess
from pathlib import Path

ROOT=Path('/app')
def smooth(x):
    x=np.clip(x,0,1)
    return x*x*(3-2*x)

def main():
    original=np.array(Image.open(ROOT/'first_frame.png').convert('RGB'))
    # Preserve the original plus, including its outline and overlapping bars.
    plus=original[602:763,115:276].copy()
    ink=np.any(plus!=255,axis=2)
    red=(plus[:,:,0]>200)&(plus[:,:,1]<40)&(plus[:,:,2]<40)
    green=original[330,420].copy()
    def clear_question(frame,cx,amount):
        ys=slice(653,712); xs=slice(cx-25,cx+26)
        area=original[ys,xs].astype(float)
        frame[ys,xs]=np.rint(area*(1-amount)+255*amount).astype(np.uint8)
    def stamp(frame,cx,cy,color,alpha=1):
        tile=plus.copy(); tile[red]=color
        y=cy-80; x=cx-80
        area=frame[y:y+161,x:x+161]
        area[ink]=np.rint(area[ink]*(1-alpha)+tile[ink]*alpha).astype(np.uint8)
    out=ROOT/'output'; out.mkdir(exist_ok=True)
    proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')],stdin=subprocess.PIPE)
    for n in range(60):
        frame=original.copy()
        if n>=5:
            clear_question(frame,482,smooth((n-5)/5))
        if n>=10:
            t=smooth((n-16)/13)
            color=np.rint(np.array([255,0,0])*(1-t)+green*t).astype(np.uint8)
            stamp(frame,482,682,color,smooth((n-10)/5))
        if n>=32:
            clear_question(frame,769,smooth((n-32)/5))
        if n>=37:
            cy=682+round(100*smooth((n-42)/13))
            stamp(frame,769,cy,green,smooth((n-37)/5))
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait()!=0: raise RuntimeError('ffmpeg failed')
if __name__=='__main__': main()
