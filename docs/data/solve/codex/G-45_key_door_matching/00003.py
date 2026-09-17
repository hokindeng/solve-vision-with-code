from pathlib import Path
from collections import deque
import numpy as np
from PIL import Image

ROOT=Path('/app')
original=np.array(Image.open(ROOT/'first_frame.png').convert('RGB'))
# The corridors occupy cells of a regular 15 by 15 grid.
centers=[int((i+.5)*1024/15) for i in range(15)]
walkable={(x,y) for y in range(15) for x in range(15) if original[centers[y],centers[x]].max()>0}
def route(start,end):
    q=deque([start]); prev={start:None}
    while q:
        p=q.popleft()
        if p==end: break
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            n=(p[0]+dx,p[1]+dy)
            if n in walkable and n not in prev:
                prev[n]=p; q.append(n)
    assert end in prev
    result=[]; p=end
    while p is not None:
        result.append((centers[p[0]],centers[p[1]])); p=prev[p]
    return result[::-1]
start=(1,1); key=(13,2); door=(9,9)
a=route(start,key); b=route(key,door)
points=np.array(a+b[1:],dtype=float)
dist=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
key_distance=dist[len(a)-1]
green=(original[:,:,1]>200)&(original[:,:,0]<50)&(original[:,:,2]<50)
yellow=(original[:,:,0]>200)&(original[:,:,1]>200)&(original[:,:,2]<50)
yy,xx=np.indices(green.shape)
key_mask=yellow&(xx>880)&(yy<220)
base=original.copy(); base[green]=255
collected=base.copy(); collected[key_mask]=255
# Translate the original circle mask to preserve its precise shape and colour.
gy,gx=np.where(green); colors=original[gy,gx]
out=ROOT/'output'; out.mkdir(exist_ok=True)
import subprocess
proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(out/'video.mp4')],stdin=subprocess.PIPE)
for frame in range(94):
    # Four initial and four final stills, with continuous movement between them.
    s=dist[-1]*np.clip((frame-3)/86,0,1)
    idx=min(np.searchsorted(dist,s,side='right')-1,len(points)-2)
    frac=(s-dist[idx])/(dist[idx+1]-dist[idx])
    pos=np.rint(points[idx]*(1-frac)+points[idx+1]*frac).astype(int)
    img=(collected if s>=key_distance else base).copy()
    dx,dy=pos-np.array(a[0])
    img[gy+dy,gx+dx]=colors
    if frame==0: assert np.array_equal(img,original)
    # Every agent pixel must remain inside the original corridor.
    assert np.all(original[gy+dy,gx+dx].max(axis=1)>0)
    proc.stdin.write(img.tobytes())
proc.stdin.close()
assert proc.wait()==0
print('Route:',len(a)-1,'steps to key,',len(b)-1,'steps to door; frames: 94')
