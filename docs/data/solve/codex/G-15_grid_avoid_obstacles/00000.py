from pathlib import Path
from collections import deque
import numpy as np
from PIL import Image
import subprocess

ROOT=Path('/app')
def main():
    original=np.array(Image.open(ROOT/'first_frame.png').convert('RGB'))
    # Coordinates are (column, row), with the top-left cell at (0, 0).
    start=(1,8); goal=(7,1)
    obstacles={(9,1),(9,3),(2,5),(6,7),(7,8),(0,9)}
    queue=deque([start]); previous={start:None}
    while queue:
        cell=queue.popleft()
        if cell==goal: break
        for dx,dy in [(0,-1),(1,0),(0,1),(-1,0)]:
            nxt=(cell[0]+dx,cell[1]+dy)
            if 0<=nxt[0]<10 and 0<=nxt[1]<10 and nxt not in obstacles and nxt not in previous:
                previous[nxt]=cell; queue.append(nxt)
    path=[]; cell=goal
    while cell is not None:
        path.append(cell); cell=previous[cell]
    path.reverse()
    assert len(path)==14
    mask=(original[:,:,0]>=190)&(original[:,:,1]>=140)&(original[:,:,1]<=205)&(original[:,:,2]==0)
    ys,xs=np.where(mask)
    colors=original[ys,xs].copy()
    background=original.copy(); background[ys,xs]=(0,100,255)
    out=ROOT/'output'; out.mkdir(exist_ok=True)
    proc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','0','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')],stdin=subprocess.PIPE)
    for i in range(62):
        if i==0:
            frame=original.copy()
        else:
            # Thirteen adjacent moves spread uniformly across the duration.
            progress=min(13.,i/60*13)
            step=min(int(progress),12); fraction=progress-step
            a=np.array(path[step],float); b=np.array(path[step+1],float)
            displacement=np.rint(((a+(b-a)*fraction)-np.array(start))*93).astype(int)
            frame=background.copy()
            frame[ys+displacement[1],xs+displacement[0]]=colors
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait()!=0: raise RuntimeError('ffmpeg failed')
if __name__=='__main__': main()
