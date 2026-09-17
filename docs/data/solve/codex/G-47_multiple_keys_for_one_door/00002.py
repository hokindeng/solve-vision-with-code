from pathlib import Path
from collections import deque
from itertools import permutations
import numpy as np
from PIL import Image
import cv2
import subprocess

ROOT=Path('/app')
image=np.array(Image.open(ROOT/'first_frame.png').convert('RGB'))
centers=[round((i+0.5)*1024/11) for i in range(11)]
walk={(r,c) for r in range(11) for c in range(11) if image[centers[r],centers[c]].max()>0}
start=(5,8)
keys=[(4,5),(1,5)]
door=(1,8)
def shortest(a,b):
    queue=deque([a]); prev={a:None}
    while queue:
        u=queue.popleft()
        if u==b: break
        for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            v=(u[0]+dr,u[1]+dc)
            if v in walk and v not in prev:
                prev[v]=u; queue.append(v)
    route=[]; u=b
    while u is not None:
        route.append(u); u=prev[u]
    return route[::-1]
choices=[]
for order in permutations(keys):
    route=[start]; arrivals={}
    for target in (*order,door):
        route+=shortest(route[-1],target)[1:]
        if target in keys: arrivals[target]=len(route)-1
    choices.append((len(route),route,arrivals,order))
_,route,arrivals,order=min(choices,key=lambda v:v[0])
print('Optimal key order:',order,'; distance:',len(route)-1,'cells')
# Preserve the exact source artwork for both stationary elements and the agent.
green=(image[:,:,1]>200)&(image[:,:,0]<50)&(image[:,:,2]<50)
cyan=(image[:,:,1]>200)&(image[:,:,2]>200)&(image[:,:,0]<50)
orange=(image[:,:,0]>200)&(image[:,:,1]>50)&(image[:,:,1]<200)&(image[:,:,2]<50)
masks={(4,5):orange,(1,5):cyan}
yg,xg=np.where(green)
colors=image[yg,xg].copy()
base=image.copy();base[green]=255
points=np.array([(centers[c],centers[r]) for r,c in route],dtype=float)
# Grid-cell interpolation gives a constant speed and follows every corridor turn.
output=ROOT/'output';output.mkdir(exist_ok=True)
proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size','1024x1024','-framerate','16','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(output/'video.mp4')],stdin=subprocess.PIPE)
for f in range(63):
    if f==0:
        frame=image.copy()
    else:
        distance=(len(route)-1)*min(f/60,1)
        k=min(int(distance),len(route)-2)
        position=points[k]+(points[k+1]-points[k])*(distance-k)
        frame=base.copy()
        for key,at in arrivals.items():
            if distance>=at:frame[masks[key]]=255
        shift=np.rint(position-points[0]).astype(int)
        frame[yg+shift[1],xg+shift[0]]=colors
    proc.stdin.write(frame.tobytes())
proc.stdin.close()
if proc.wait()!=0:raise RuntimeError('ffmpeg failed')
