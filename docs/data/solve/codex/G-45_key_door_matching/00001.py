from pathlib import Path
from collections import deque
import numpy as np
from PIL import Image
import cv2
import subprocess

ROOT=Path('/app')
OUT=ROOT/'output'
OUT.mkdir(exist_ok=True)
im=np.array(Image.open(ROOT/'first_frame.png').convert('RGB'))
green=(im[:,:,1]>200)&(im[:,:,0]<20)&(im[:,:,2]<20)
yellow=(im[:,:,0]>200)&(im[:,:,1]>200)&(im[:,:,2]<20)
keymask=yellow & (np.indices(yellow.shape)[0]>600)
gy,gx=np.where(green)
ky,kx=np.where(keymask)
start=(1,1); key=(13,10); door=(9,1)
def center(p): return np.array([(p[0]+.5)*1024/15,(p[1]+.5)*1024/15])
walk=set()
for y in range(15):
 for x in range(15):
  px,py=np.rint(center((x,y))).astype(int)
  if im[py,px].max()>0: walk.add((x,y))
def route(a,b):
 q=deque([a]); prev={a:None}
 while q:
  u=q.popleft()
  if u==b: break
  for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
   v=(u[0]+dx,u[1]+dy)
   if v in walk and v not in prev: prev[v]=u;q.append(v)
 path=[];u=b
 while u is not None: path.append(u);u=prev[u]
 return path[::-1]
p1=route(start,key);p2=route(key,door)
points=np.array([center(p) for p in p1+p2[1:]])
origin=np.array([gx.mean(),gy.mean()])
points[0]=origin
points[len(p1)-1]=[kx.mean(),ky.mean()]
lengths=np.linalg.norm(np.diff(points,axis=0),axis=1)
cumulative=np.r_[0,np.cumsum(lengths)]
collect=cumulative[len(p1)-1]
base=im.copy();base[green]=255
encoder=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','0','-preset','medium','-pix_fmt','yuv420p',str(OUT/'video.mp4')],stdin=subprocess.PIPE)
# Four frames at the start and six at the destination; steady movement between.
for f in range(154):
 if f==0: frame=im.copy()
 else:
  distance=np.clip((f-3)/144,0,1)*cumulative[-1]
  segment=min(np.searchsorted(cumulative,distance,side='right')-1,len(lengths)-1)
  t=(distance-cumulative[segment])/lengths[segment]
  pos=points[segment]*(1-t)+points[segment+1]*t
  frame=base.copy()
  if distance>=collect: frame[keymask]=255
  dx,dy=np.rint(pos-origin).astype(int)
  frame[gy+dy,gx+dx]=im[gy,gx]
 encoder.stdin.write(frame.tobytes())
encoder.stdin.close()
assert encoder.wait()==0
print('Route to key:',p1)
print('Route to door:',p2)
