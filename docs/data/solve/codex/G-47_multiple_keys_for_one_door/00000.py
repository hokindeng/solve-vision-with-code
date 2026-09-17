from pathlib import Path
from collections import deque
from itertools import permutations
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')
OUT = ROOT/'output'
OUT.mkdir(exist_ok=True)
im = np.array(Image.open(ROOT/'first_frame.png').convert('RGB'))
green = (im[:,:,1]>200)&(im[:,:,0]<50)&(im[:,:,2]<50)
orange = (im[:,:,0]>200)&(im[:,:,1]>60)&(im[:,:,1]<200)&(im[:,:,2]<50)
blue = (im[:,:,2]>200)&(im[:,:,0]<50)&(im[:,:,1]<50)
door = (im[:,:,0]>200)&(im[:,:,2]>200)&(im[:,:,1]<50)
def center(mask):
    y,x=np.where(mask)
    return (float(x.mean()),float(y.mean()))
def cell(mask):
    x,y=center(mask)
    return (int(x*11/1024),int(y*11/1024))
def xy(c):
    return np.array([(c[0]+.5)*1024/11,(c[1]+.5)*1024/11])
start=cell(green)
keys=[cell(orange),cell(blue)]
goal=cell(door)
walk=set()
for y in range(11):
    for x in range(11):
        px,py=np.rint(xy((x,y))).astype(int)
        if im[py,px].max()>0:
            walk.add((x,y))
def bfs(a,b):
    q=deque([a]); prev={a:None}
    while q:
        v=q.popleft()
        if v==b:break
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            u=(v[0]+dx,v[1]+dy)
            if u in walk and u not in prev:
                prev[u]=v;q.append(u)
    p=[b]
    while p[-1]!=a:p.append(prev[p[-1]])
    return p[::-1]
def route(order):
    nodes=[start,*order,goal]
    p=[start];hits={}
    for a,b in zip(nodes,nodes[1:]):
        p.extend(bfs(a,b)[1:])
        if b in keys:hits[b]=len(p)-1
    return p,hits
order=min(permutations(keys),key=lambda o:len(route(o)[0]))
path,hits=route(order)
print('Optimal order:',order,'route length:',len(path)-1,'cells')
# Restore only the pixels occupied by the initial agent and collected keys.
base=im.copy();base[green]=255
ys,xs=np.where(green)
sprite=im[ys,xs].copy()
cx,cy=center(green)
points=np.array([xy(c) for c in path])
points[0]=[cx,cy]
# Exact symbol centers for pickup and final position.
for mask in [orange,blue,door]:
    c=cell(mask)
    for i,v in enumerate(path):
        if v==c:points[i]=center(mask)
lengths=np.linalg.norm(np.diff(points,axis=0),axis=1)
cumulative=np.r_[0,np.cumsum(lengths)]
cmd=['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-preset','slow','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')]
p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
for f in range(51):
    if f==0:
        frame=im.copy()
    else:
        d=cumulative[-1]*min(f/49,1)
        j=min(np.searchsorted(cumulative,d,side='right')-1,len(points)-2)
        t=(d-cumulative[j])/lengths[j]
        pos=points[j]*(1-t)+points[j+1]*t
        frame=base.copy()
        for k,mask in zip(keys,[orange,blue]):
            if d>=cumulative[hits[k]]:frame[mask]=255
        dx,dy=np.rint(pos-[cx,cy]).astype(int)
        frame[ys+dy,xs+dx]=sprite
    p.stdin.write(frame.tobytes())
p.stdin.close()
err=p.stderr.read();code=p.wait()
if code:raise RuntimeError(err.decode())
print(OUT/'video.mp4')
