from PIL import Image
import numpy as np
import cv2
import subprocess
from pathlib import Path

ROOT=Path('/app')
costs=[0,50,20,40,30,30,10,40,10,30,50,10,10,20,10,20]
best_score=-1
best_path=[]
def search(cell,seen,path,score):
    global best_score,best_path
    if cell==15:
        if score>best_score:
            best_score,best_path=score,path[:]
        return
    r,c=divmod(cell,4)
    for dr,dc in [(0,1),(1,0),(0,-1),(-1,0)]:
        rr,cc=r+dr,c+dc
        if 0<=rr<4 and 0<=cc<4:
            nxt=rr*4+cc
            if not (seen>>nxt)&1:
                search(nxt,seen|(1<<nxt),path+[nxt],score+costs[nxt])
search(0,1,[0],0)
original=np.array(Image.open(ROOT/'first_frame.png').convert('RGB'))
background=original.copy()
green=original[20,20].copy()
# Extract the original character, including its white mouth and black outline.
region=original[60:230,45:215]
mask=np.any(region!=green,axis=2)
yy,xx=np.where(mask)
yy=yy+60
xx=xx+45
pixels=original[yy,xx].copy()
background[yy,xx]=green
out=ROOT/'output'
out.mkdir(exist_ok=True)
proc=subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-preset','slow','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')],stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
for frame in range(91):
    if frame==0:
        image=original.copy()
    else:
        # Spend six frames on every step, then hold on the goal.
        t=min(1,max(0,(frame-2)/84))*(len(best_path)-1)
        i=min(int(t),len(best_path)-2)
        f=t-i
        a,b=best_path[i:i+2]
        ar,ac=divmod(a,4)
        br,bc=divmod(b,4)
        dx=round(256*(ac+(bc-ac)*f))
        dy=round(256*(ar+(br-ar)*f))
        image=background.copy()
        image[yy+dy,xx+dx]=pixels
    proc.stdin.write(image.tobytes())
proc.stdin.close()
if proc.wait()!=0:
    raise RuntimeError('ffmpeg encoding failed')
print('Maximum simple-path cost:',best_score)
print('Path:',[(v//4+1,v%4+1) for v in best_path])
