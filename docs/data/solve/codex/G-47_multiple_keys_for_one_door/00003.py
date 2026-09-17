from pathlib import Path
from collections import deque
from itertools import permutations
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    green = np.all(original == (0,255,0), axis=2)
    key_masks = [np.all(original == c, axis=2) for c in [(0,0,255),(255,0,255)]]
    ys,xs = np.where(green)
    start_xy = np.array([xs.mean(), ys.mean()])
    def cell(xy):
        return tuple(np.rint((np.array(xy)-118.125)/78.75).astype(int))
    def center(c):
        return np.array(c)*78.75+118.125
    start=cell(start_xy)
    keys=[]
    for mask in key_masks:
        y,x=np.where(mask)
        keys.append(cell([x.mean(),y.mean()]))
    yellow=np.all(original==(255,255,0),axis=2)
    y,x=np.where(yellow)
    door=cell([x.mean(),y.mean()])
    walkable=set()
    for j in range(11):
        for i in range(11):
            x,y=np.rint(center((i,j))).astype(int)
            if np.any(original[y,x]): walkable.add((i,j))
    def shortest(a,b):
        q=deque([a]); prev={a:None}
        while q:
            p=q.popleft()
            if p==b: break
            for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                n=(p[0]+dx,p[1]+dy)
                if n in walkable and n not in prev:
                    prev[n]=p;q.append(n)
        path=[b]
        while path[-1]!=a: path.append(prev[path[-1]])
        return path[::-1]
    best=None
    for order in permutations(range(len(keys))):
        route=[start]; arrivals={}
        for idx in order:
            route+=shortest(route[-1],keys[idx])[1:]
            arrivals[idx]=len(route)-1
        route+=shortest(route[-1],door)[1:]
        if best is None or len(route)<len(best[0]): best=(route,arrivals,order)
    route,arrivals,order=best
    print('Optimal key order:', ['blue' if k==0 else 'magenta' for k in order], 'Distance:',len(route)-1,'cells')
    points=np.array([center(c) for c in route])
    points[0]=start_xy
    clean=original.copy();clean[green]=255
    (ROOT/'output').mkdir(exist_ok=True)
    cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','0','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(ROOT/'output/video.mp4')]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    for f in range(59):
        progress=(len(route)-1)*f/58
        seg=min(int(progress),len(route)-2)
        pos=points[seg]+(points[seg+1]-points[seg])*(progress-seg)
        frame=clean.copy()
        for idx,at in arrivals.items():
            if progress>=at: frame[key_masks[idx]]=255
        shift=np.rint(pos-start_xy).astype(int)
        frame[ys+shift[1],xs+shift[0]]=(0,255,0)
        if f==0: frame=original
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait()!=0: raise RuntimeError('ffmpeg failed')

if __name__=='__main__': main()
