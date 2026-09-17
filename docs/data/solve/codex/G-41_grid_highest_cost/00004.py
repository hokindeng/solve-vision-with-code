from PIL import Image
import numpy as np
import subprocess, os

ROOT='/app'
COSTS=[0,50,10,30,20,10,50,20,50,30,10,30,20,10,10,50]
# A maximum simple path: unrestricted revisits have no finite maximum.
best_score=-1
best_path=[]
def search(cell, mask, score, path):
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
            if not (mask>>nxt)&1:
                search(nxt,mask|1<<nxt,score+COSTS[nxt],path+[nxt])
search(0,1,0,[0])

def main():
    original=Image.open(ROOT+'/first_frame.png').convert('RGB')
    pixels=np.array(original)
    green=pixels[20,20]
    # Isolate the original character, including its white mouth and black outline.
    patch=pixels[70:225,50:205].copy()
    mask=np.any(patch!=green,axis=2)
    rgba=np.concatenate((patch,(mask.astype(np.uint8)*255)[...,None]),axis=2)
    sprite=Image.fromarray(rgba,'RGBA')
    background=pixels.copy()
    region=background[70:225,50:205]
    region[mask]=green
    base=Image.fromarray(background)
    os.makedirs(ROOT+'/output',exist_ok=True)
    proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','15','-pix_fmt','yuv420p','-movflags','+faststart',ROOT+'/output/video.mp4'],stdin=subprocess.PIPE)
    for frame in range(91):
        if frame==0:
            im=original
        else:
            progress=min(1,max(0,(frame-3)/83))*(len(best_path)-1)
            leg=min(int(progress),len(best_path)-2)
            fraction=min(1,progress-leg)
            # Brief stop at every cell makes the orthogonal steps readable.
            fraction=min(1,fraction/0.8)
            a,b=best_path[leg],best_path[leg+1]
            ar,ac=divmod(a,4); br,bc=divmod(b,4)
            x=round(50+256*(ac+(bc-ac)*fraction))
            y=round(70+256*(ar+(br-ar)*fraction))
            im=base.copy()
            im.paste(sprite,(x,y),sprite)
        proc.stdin.write(np.asarray(im).tobytes())
    proc.stdin.close()
    if proc.wait(): raise RuntimeError('ffmpeg failed')
    print('Maximum simple path cost:',best_score)
    print('Path (row, column):',[(v//4+1,v%4+1) for v in best_path])
if __name__=='__main__': main()
