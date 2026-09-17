from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT=Path('/app')

def main():
    src=np.array(Image.open(ROOT/'first_frame.png').convert('RGB'))
    cyan=np.all(src==[0,191,255],axis=2)
    orange=np.all(src==[255,140,0],axis=2)
    yy,xx=np.indices(cyan.shape)
    # Recover the original circle geometry, including the portion under the arrow.
    best=None
    for cx in np.arange(515,516.01,.1):
        for cy in np.arange(893,894.01,.1):
            for radius in np.arange(30,30.81,.1):
                disk=(xx[860:927,482:549]-cx)**2+(yy[860:927,482:549]-cy)**2<=radius**2
                error=np.count_nonzero((disk!=cyan[860:927,482:549])&~orange[860:927,482:549])
                if best is None or error<best[0]:best=(error,cx,cy,radius)
    _,cx,cy,r=best
    background=src.copy()
    background[cyan|orange]=255
    start=np.array([cx,cy])
    direction=np.array([128.,-91.])
    direction/=np.linalg.norm(direction)
    # Interior walls occupy [63,962); account for the ball's radius.
    right=962-r
    top=63+r
    d1=(right-cx)/direction[0]
    contact1=start+d1*direction
    reflected=direction*np.array([-1.,1.])
    d2=(top-contact1[1])/reflected[1]
    contact2=contact1+d2*reflected
    out=ROOT/'output'
    out.mkdir(exist_ok=True)
    cmd=['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','17','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
    for i in range(80):
        if i<4:
            frame=src
        else:
            distance=np.clip((i-4)/72,0,1)*(d1+d2)
            pos=start+direction*distance if distance<=d1 else contact1+reflected*(distance-d1)
            frame=background.copy()
            disk=(xx-pos[0])**2+(yy-pos[1])**2<=r*r
            frame[disk]=[0,191,255]
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait()!=0:raise RuntimeError('Video encoding failed')
    print(f'Circle: {cx:.1f}, {cy:.1f}, radius {r:.1f}; contacts: {contact1}, {contact2}')

if __name__=='__main__':main()
