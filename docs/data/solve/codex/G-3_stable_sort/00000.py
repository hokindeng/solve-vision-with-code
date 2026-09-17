from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import subprocess

ROOT=Path('/app')
def ease(t):
    t=np.clip(t,0,1)
    return t*t*(3-2*t)

def main():
    src=np.array(Image.open(ROOT/'first_frame.png').convert('RGB'))
    bg=src[0,0]
    mask=np.any(src!=bg,axis=2).astype(np.uint8)
    n,labels,stats,cents=cv2.connectedComponentsWithStats(mask,8)
    shapes=[]
    for k in range(1,n):
        x,y,w,h,area=stats[k]
        cut=src[y:y+h,x:x+w].copy()
        m=labels[y:y+h,x:x+w]==k
        typ=0 if np.any(np.all(cut==[120,220,170],axis=2)) else 1
        shapes.append(dict(cut=cut,mask=m,w=w,h=h,typ=typ,start=np.array([x+(w-1)/2,y+(h-1)/2],float)))
    for typ in (0,1):
        group=sorted([s for s in shapes if s['typ']==typ],key=lambda s:s['start'][0])
        for i,s in enumerate(group):
            s['group']=np.array([220+290*i,330 if typ==0 else 700],float)
        for i,s in enumerate(sorted(group,key=lambda s:s['w'])):
            s['rank']=i
            s['sorted']=np.array([220+290*i,330 if typ==0 else 700],float)
    ordered=sorted(shapes,key=lambda s:(s['typ'],s['w']))
    gap=25
    left=(1024-sum(s['w'] for s in ordered)-gap*5)/2
    for s in ordered:
        s['final']=np.array([left+(s['w']-1)/2,512.])
        left+=s['w']+gap
    out=ROOT/'output';out.mkdir(exist_ok=True)
    proc=subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')],stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
    for f in range(96):
        frame=np.full_like(src,bg)
        for s in shapes:
            if f<=28:
                t=ease(f/28); p=(1-t)*s['start']+t*s['group']
            elif f<=68:
                u=(f-28)/40;t=ease(u);p=(1-t)*s['group']+t*s['sorted']
                # Passing lanes keep the original shapes separate while sorting.
                if s['rank']==0:p[1]-=145*np.sin(np.pi*t)
                elif s['rank']==2:p[1]+=130*np.sin(np.pi*t)
            else:
                t=ease((f-68)/27);p=(1-t)*s['sorted']+t*s['final']
            x=int(round(p[0]-(s['w']-1)/2));y=int(round(p[1]-(s['h']-1)/2))
            region=frame[y:y+s['h'],x:x+s['w']]
            region[s['mask']]=s['cut'][s['mask']]
        if f==0:assert np.array_equal(frame,src)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait()!=0:raise RuntimeError('ffmpeg failed')
if __name__=='__main__':main()
