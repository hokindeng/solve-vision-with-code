from pathlib import Path
from collections import deque
import numpy as np
from PIL import Image, ImageDraw
import cv2

ROOT=Path('/app')
source=Image.open(ROOT/'first_frame.png').convert('RGB')
a=np.array(source)
N=15
step=1024/N
start=(3,5)
end=(13,13)
def center(cell):
    r,c=cell
    return ((c+.5)*step,(r+.5)*step)
open_cells=set()
for r in range(N):
    for c in range(N):
        x,y=center((r,c))
        # Most pixels in the central square remain white even beneath a marker.
        patch=a[int(y)-24:int(y)+25,int(x)-24:int(x)+25]
        if np.mean(np.all(patch>230,axis=2))>.25:
            open_cells.add((r,c))
queue=deque([start]); previous={start:None}
while queue:
    u=queue.popleft()
    if u==end: break
    for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
        v=(u[0]+dr,u[1]+dc)
        if v in open_cells and v not in previous:
            previous[v]=u
            queue.append(v)
route=[]
u=end
while u is not None:
    route.append(u)
    u=previous[u]
route.reverse()
assert all(v in open_cells for v in route)
points=[center(cell) for cell in route]
# Restore the white pathway under the moving green starting marker.
base=a.copy()
green=(a[:,:,1]>120)&(a[:,:,0]<100)&(a[:,:,2]<100)
base[green]=255
# Keep the flag and pole at the destination exactly as supplied.
flag=np.zeros(a.shape[:2],dtype=bool)
flag[890:951,895:952]=np.any(a[890:951,895:952]<230,axis=2)
ROOT.joinpath('output').mkdir(exist_ok=True)
raw=ROOT/'output'/'temp.mp4'
writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),16,(1024,1024))
for frame in range(58):
    if frame==0:
        result=a.copy()
    else:
        t=(frame/57)*(len(points)-1)
        index=min(int(t),len(points)-2)
        f=t-index
        x=points[index][0]*(1-f)+points[index+1][0]*f
        y=points[index][1]*(1-f)+points[index+1][1]*f
        im=Image.fromarray(base)
        draw=ImageDraw.Draw(im)
        traveled=points[:index+1]+[(x,y)]
        draw.line(traveled,fill=(112,218,108),width=15,joint='curve')
        for px,py in traveled:
            draw.ellipse((px-7,py-7,px+7,py+7),fill=(112,218,108))
        draw.ellipse((x-20,y-20,x+20,y+20),fill=(43,197,43))
        result=np.array(im)
        result[flag]=a[flag]
    writer.write(cv2.cvtColor(result,cv2.COLOR_RGB2BGR))
writer.release()
import subprocess
subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(raw),'-c:v','libx264','-crf','15','-pix_fmt','yuv420p','-r','16',str(ROOT/'output'/'video.mp4')],check=True)
raw.unlink()
print(f'Animated {len(route)} cells in 58 frames.')
