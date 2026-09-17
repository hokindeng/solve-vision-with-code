from pathlib import Path
from collections import deque
import subprocess
import numpy as np
from PIL import Image

ROOT = Path('/app')
FPS, N = 16, 282

def main():
    original = np.array(Image.open(ROOT/'first_frame.png').convert('RGB'))
    green = (original[:,:,1] > 200) & (original[:,:,0] < 40) & (original[:,:,2] < 40)
    red_key = (original[:,:,0] > 200) & (original[:,:,1] < 40) & (original[:,:,2] < 40)
    red_key[:580] = False
    gy, gx = np.where(green)
    cx, cy = int(round(gx.mean())), int(round(gy.mean()))
    offsets = np.column_stack((gx-cx, gy-cy))
    background = original.copy()
    background[green] = 255
    walkable = np.any(original > 0, axis=2)
    def route(start, end):
        parents = {start:None}
        q = deque([start])
        while q:
            p = q.popleft()
            if p == end: break
            for dx,dy in [(0,1),(1,0),(0,-1),(-1,0)]:
                n = (p[0]+dx,p[1]+dy)
                if not (0 <= n[0] < 17 and 0 <= n[1] < 17) or n in parents: continue
                x,y = 80+54*n[0],80+54*n[1]
                px,py = 80+54*p[0],80+54*p[1]
                if walkable[y,x] and walkable[(y+py)//2,(x+px)//2]:
                    parents[n] = p
                    q.append(n)
        assert end in parents
        out=[]
        p=end
        while p is not None:
            out.append((80+54*p[0],80+54*p[1]))
            p=parents[p]
        return np.array(out[::-1], dtype=float)
    first = route((0,0),(4,10))
    second = route((4,10),(10,0))
    first[0] = [cx,cy]
    def lengths(points):
        return np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    l1,l2=lengths(first), lengths(second)
    # Allocate travel time by distance to keep the agent's speed constant.
    start_hold,key_hold,end_hold=6,7,12
    travel=N-1-start_hold-key_hold-end_hold
    travel1=round(travel*l1[-1]/(l1[-1]+l2[-1]))
    travel2=travel-travel1
    pickup=start_hold+travel1
    def position(points, ls, fraction):
        d=np.clip(fraction,0,1)*ls[-1]
        i=min(np.searchsorted(ls,d,side='right')-1,len(points)-2)
        return np.rint(points[i]+(points[i+1]-points[i])*(d-ls[i])/(ls[i+1]-ls[i])).astype(int)
    (ROOT/'output').mkdir(exist_ok=True)
    ff=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r',str(FPS),'-i','-','-an','-c:v','libx264','-crf','18','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(ROOT/'output/video.mp4')],stdin=subprocess.PIPE)
    for f in range(N):
        frame=background.copy()
        if f < pickup:
            p=position(first,l1,(f-start_hold)/travel1)
        else:
            frame[red_key]=255
            p=position(second,l2,(f-pickup-key_hold)/travel2)
        xs=offsets[:,0]+p[0]
        ys=offsets[:,1]+p[1]
        # Every agent pixel must stay in a white corridor.
        assert np.all(walkable[ys,xs])
        frame[ys,xs]=original[gy,gx]
        if f==0:
            assert np.array_equal(frame,original)
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    assert ff.wait()==0
    print(f'Created {N} frames; collected red key at frame {pickup}; arrived at frame {N-1-end_hold}.')

if __name__=='__main__':
    main()
