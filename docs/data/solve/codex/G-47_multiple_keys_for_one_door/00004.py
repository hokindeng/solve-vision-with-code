from pathlib import Path
from collections import deque
from itertools import permutations
import numpy as np
from PIL import Image
import subprocess

ROOT=Path('/app')
a=np.array(Image.open(ROOT/'first_frame.png').convert('RGB'))
mask_green=(a[:,:,1]>200)&(a[:,:,0]<50)&(a[:,:,2]<50)
mask_red=(a[:,:,0]>200)&(a[:,:,1]<50)&(a[:,:,2]<50)
mask_cyan=(a[:,:,0]<50)&(a[:,:,1]>200)&(a[:,:,2]>200)
def center(mask):
 y,x=np.where(mask); return np.array([x.mean(),y.mean()])
start=(7,10); door=(3,5)
keys={(9,3):mask_red,(6,1):mask_cyan}
walk=set()
for r in range(13):
 for c in range(13):
  if a[int((r+.5)*1024/13),int((c+.5)*1024/13)].max()>0: walk.add((r,c))
def route(s,t):
 q=deque([s]); prev={s:None}
 while q:
  v=q.popleft()
  if v==t: break
  for d in [(0,1),(1,0),(0,-1),(-1,0)]:
   w=(v[0]+d[0],v[1]+d[1])
   if w in walk and w not in prev: prev[w]=v;q.append(w)
 p=[t]
 while p[-1]!=s:p.append(prev[p[-1]])
 return p[::-1]
def length(order):
 nodes=[start,*order,door]
 return sum(len(route(s,t))-1 for s,t in zip(nodes,nodes[1:]))
order=min(permutations(keys),key=length)
print('Optimal key order:',order,'distance:',length(order),'cells')
print('Alternative distances:',[(o,length(o)) for o in permutations(keys)])
centers={v:np.array([(v[1]+.5)*1024/13,(v[0]+.5)*1024/13]) for v in walk}
centers[start]=center(mask_green)
for k,m in keys.items():centers[k]=center(m)
# Preserve the original agent silhouette and color exactly by translating its pixels.
gy,gx=np.where(mask_green); origin=center(mask_green)
base=a.copy();base[mask_green]=255
nodes=[start,*order,door]
paths=[route(s,t) for s,t in zip(nodes,nodes[1:])]
lengths=np.array([len(p)-1 for p in paths])
ends=np.rint(np.cumsum(lengths)/sum(lengths)*72).astype(int)
frames=[a.copy()]; picked=[]; first=0
for j,path in enumerate(paths):
 points=np.array([centers[v] for v in path]);ds=np.linalg.norm(np.diff(points,axis=0),axis=1)
 cumulative=np.r_[0,np.cumsum(ds)]
 count=ends[j]-first
 for step in range(1,count+1):
  dist=step/count*cumulative[-1]
  idx=min(np.searchsorted(cumulative,dist,side='right')-1,len(ds)-1)
  pos=points[idx]+(points[idx+1]-points[idx])*(dist-cumulative[idx])/ds[idx]
  frame=base.copy()
  for k in picked:frame[keys[k]]=255
  if step==count and j<len(order):frame[keys[order[j]]]=255
  dx,dy=np.rint(pos-origin).astype(int)
  frame[gy+dy,gx+dx]=a[gy,gx]
  frames.append(frame)
 if j<len(order):picked.append(order[j])
 first=ends[j]
assert len(frames)==73
# The moving agent must stay wholly inside the white maze corridors.
black=np.all(a==0,axis=2)
for frame in frames:
 green=(frame[:,:,1]>200)&(frame[:,:,0]<50)&(frame[:,:,2]<50)
 assert not np.any(green&black)
 assert np.array_equal(frame[black],a[black])
(ROOT/'output').mkdir(exist_ok=True)
proc=subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','15','-pix_fmt','yuv420p','-movflags','+faststart',str(ROOT/'output/video.mp4')],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
_,err=proc.communicate(b''.join(f.tobytes() for f in frames))
if proc.returncode:raise RuntimeError(err.decode())
print('Saved',ROOT/'output/video.mp4')
