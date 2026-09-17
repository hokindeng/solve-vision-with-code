from pathlib import Path
from collections import deque
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT=Path('/app')
OUT=ROOT/'output'
OUT.mkdir(exist_ok=True)
base=Image.open(ROOT/'first_frame.png').convert('RGB')
arr=np.asarray(base)
N=15
centers=[(i+.5)*1024/N for i in range(N)]
walk=np.array([[arr[int(centers[r]),int(centers[c])].max()>80 for c in range(N)] for r in range(N)])
start=(2,3)
end=(10,14)
walk[start]=walk[end]=True
queue=deque([start])
parent={start:None}
while queue:
    u=queue.popleft()
    if u==end: break
    for dr,dc in [(0,1),(1,0),(0,-1),(-1,0)]:
        v=(u[0]+dr,u[1]+dc)
        if 0<=v[0]<N and 0<=v[1]<N and walk[v] and v not in parent:
            parent[v]=u
            queue.append(v)
assert end in parent
path=[]
u=end
while u is not None:
    path.append(u)
    u=parent[u]
path.reverse()
assert all(abs(a[0]-b[0])+abs(a[1]-b[1])==1 for a,b in zip(path,path[1:]))
points=[(centers[c],centers[r]) for r,c in path]
# Preserve the original start marker and flag over the added route.
y,x=np.indices(arr.shape[:2])
markers=((arr[:,:,1]>100)&(arr[:,:,0]<80)&(arr[:,:,2]<100)) | ((x>965)&(y>685)&(y<747)&(np.min(arr,axis=2)<230))
proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(OUT/'video.mp4')],stdin=subprocess.PIPE)
for frame in range(71):
    canvas=base.copy()
    if frame:
        progress=(len(points)-1)*frame/70
        i=min(int(progress),len(points)-2)
        t=progress-i
        tip=(points[i][0]+(points[i+1][0]-points[i][0])*t,points[i][1]+(points[i+1][1]-points[i][1])*t)
        visited=points[:i+1]+[tip]
        draw=ImageDraw.Draw(canvas)
        draw.line(visited,fill=(44,198,43),width=12,joint='curve')
        for px,py in visited:
            draw.ellipse((px-6,py-6,px+6,py+6),fill=(44,198,43))
        px,py=tip
        draw.ellipse((px-18,py-18,px+18,py+18),fill=(44,198,43))
        data=np.asarray(canvas).copy()
        data[markers]=arr[markers]
    else:
        data=arr
    proc.stdin.write(data.tobytes())
proc.stdin.close()
assert proc.wait()==0
print(f'Created {OUT / "video.mp4"}; route: {len(path)} cells, 71 frames.')
