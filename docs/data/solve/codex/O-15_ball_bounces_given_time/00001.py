from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT=Path('/app')
def main():
    original=np.array(Image.open(ROOT/'first_frame.png').convert('RGB'))
    background=original.copy()
    ball_color=np.array([210,105,30],dtype=np.uint8)
    moving=np.all(original==ball_color,axis=2)|np.all(original==[255,140,0],axis=2)
    background[moving]=255
    # Inner wall faces are x/y = 63 and 962. The ball radius is 30.
    lo,hi=93.0,932.0
    p=np.array([765.0,280.0])
    v=np.array([160.0,-6.0])
    segments=[]
    total=0.0
    for _ in range(5):
        times=np.where(v>0,(hi-p)/v,(lo-p)/v)
        dt=float(np.min(times))
        q=p+dt*v
        segments.append((total,total+dt,p.copy(),v.copy()))
        total+=dt
        hit=np.isclose(times,dt)
        v[hit]*=-1
        p=q
    final=p.copy()
    out=ROOT/'output'
    out.mkdir(exist_ok=True)
    cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','15','-preset','slow','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    yy,xx=np.ogrid[:1024,:1024]
    for i in range(80):
        if i==0:
            frame=original
        else:
            t=total*min(i/78.0,1.0)
            center=final
            for start,end,pos,vel in segments:
                if t<=end:
                    center=pos+(t-start)*vel
                    break
            frame=background.copy()
            mask=(xx-center[0])**2+(yy-center[1])**2<=30.5**2
            frame[mask]=ball_color
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait()!=0:
        raise RuntimeError('ffmpeg failed')
    print('Five collisions completed; final center:',final.tolist())
if __name__=='__main__':
    main()
