from pathlib import Path
from collections import deque
from itertools import permutations
import numpy as np
from PIL import Image
import cv2
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
im = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
# The maze consists of 13 equal square cells in each direction.
centers = [int((i + .5) * 1024 / 13) for i in range(13)]
walkable = {(x,y) for y in range(13) for x in range(13)
            if im[centers[y], centers[x]].max() > 0}
green = (im[:,:,1]>200)&(im[:,:,0]<50)&(im[:,:,2]<50)
key_masks = [(im[:,:,2]>200)&(im[:,:,0]<50)&(im[:,:,1]<50),
             (im[:,:,0]>200)&(im[:,:,2]>200)&(im[:,:,1]<50)]
def cell_for(mask):
    yy,xx=np.where(mask)
    return (int(xx.mean()*13/1024),int(yy.mean()*13/1024))
start=cell_for(green)
keys=[cell_for(m) for m in key_masks]
door=cell_for((im[:,:,0]>200)&(im[:,:,1]>200)&(im[:,:,2]<50))
def shortest(a,b):
    queue=deque([a]); prev={a:None}
    while queue:
        p=queue.popleft()
        if p==b: break
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            q=(p[0]+dx,p[1]+dy)
            if q in walkable and q not in prev:
                prev[q]=p; queue.append(q)
    route=[b]
    while route[-1]!=a: route.append(prev[route[-1]])
    return route[::-1]
def length(order):
    stops=[start]+list(order)+[door]
    return sum(len(shortest(a,b))-1 for a,b in zip(stops,stops[1:]))
order=min(permutations(keys),key=length)
route=[start]; pickups={}
for stop in list(order)+[door]:
    route.extend(shortest(route[-1],stop)[1:])
    if stop in keys: pickups[keys.index(stop)]=len(route)-1
points=np.array([(centers[x],centers[y]) for x,y in route],float)
# Use the original circle pixels, retaining its exact size and rasterization.
gy,gx=np.where(green)
base=im.copy(); base[green]=255
frames=[]
for i in range(49):
    progress=i*(len(route)-1)/48
    j=min(int(progress),len(route)-2)
    pos=points[j]+(points[j+1]-points[j])*(progress-j)
    frame=base.copy()
    for k,at in pickups.items():
        if progress>=at: frame[key_masks[k]]=255
    dx,dy=np.rint(pos-points[0]).astype(int)
    frame[gy+dy,gx+dx]=im[gy,gx]
    if i==0: frame=im.copy()
    frames.append(frame)
proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
    '-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','18',
    '-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')],stdin=subprocess.PIPE)
for frame in frames: proc.stdin.write(frame.tobytes())
proc.stdin.close()
if proc.wait()!=0: raise RuntimeError('ffmpeg failed')
print('Key order:',order,'; route length:',len(route)-1,'cells; 49 frames')
