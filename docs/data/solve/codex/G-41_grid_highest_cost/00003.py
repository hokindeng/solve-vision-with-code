from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path('/app')
COSTS = [[0,10,40,30],[40,30,30,20],[20,10,40,30],[50,30,10,40]]

def best_simple_path():
    # With positive costs, unrestricted revisits have no finite maximum.
    # Maximize the collected cost among paths without repeated cells.
    best_score, best_path = -1, []
    def visit(r,c,seen,path,score):
        nonlocal best_score,best_path
        if (r,c)==(3,3):
            if score>best_score:
                best_score,best_path=score,path[:]
            return
        for dr,dc in [(0,1),(1,0),(0,-1),(-1,0)]:
            nr,nc=r+dr,c+dc
            if 0<=nr<4 and 0<=nc<4 and (nr,nc) not in seen:
                visit(nr,nc,seen|{(nr,nc)},path+[(nr,nc)],score+COSTS[nr][nc])
    visit(0,0,{(0,0)},[(0,0)],0)
    return best_score,best_path

def main():
    original=np.asarray(Image.open(ROOT/'first_frame.png').convert('RGB')).copy()
    green=np.array([76,175,80],dtype=np.uint8)
    # Extract only the existing character; all grid pixels remain in the base.
    mask=np.zeros((1024,1024),dtype=bool)
    mask[4:252,4:252]=np.any(original[4:252,4:252]!=green,axis=2)
    sy,sx=np.where(mask)
    pixels=original[sy,sx].copy()
    base=original.copy()
    base[sy,sx]=green
    score,path=best_simple_path()
    (ROOT/'output').mkdir(exist_ok=True)
    command=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24',
             '-video_size','1024x1024','-framerate','16','-i','-',
             '-an','-c:v','libx264','-crf','18','-preset','medium','-pix_fmt','yuv420p',
             '-movflags','+faststart',str(ROOT/'output/video.mp4')]
    proc=subprocess.Popen(command,stdin=subprocess.PIPE)
    for frame in range(91):
        t=min(frame/86,1)*(len(path)-1)
        step=min(int(t),len(path)-2)
        fraction=t-step
        r=path[step][0]+fraction*(path[step+1][0]-path[step][0])
        c=path[step][1]+fraction*(path[step+1][1]-path[step][1])
        rendered=base.copy()
        rendered[sy+round(r*256),sx+round(c*256)]=pixels
        if frame==0:
            assert np.array_equal(rendered,original)
        proc.stdin.write(rendered.tobytes())
    proc.stdin.close()
    if proc.wait()!=0:
        raise RuntimeError('ffmpeg failed')
    print('Maximum simple-path cost:',score)
    print('Path (row, column):',path)

if __name__=='__main__':
    main()
