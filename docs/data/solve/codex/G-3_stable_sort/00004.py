from pathlib import Path
import cv2
import numpy as np
import subprocess

ROOT=Path('/app')
OUT=ROOT/'output'
OUT.mkdir(exist_ok=True)
original=cv2.imread(str(ROOT/'first_frame.png'))
bg=original[0,0].copy()
mask=np.any(original!=bg,axis=2).astype(np.uint8)
n,labels,stats,_=cv2.connectedComponentsWithStats(mask)
shapes=[]
for k,(x,y,w,h,area) in enumerate(stats[1:],1):
    shapes.append(dict(pixels=original[y:y+h,x:x+w].copy(),mask=labels[y:y+h,x:x+w]==k,w=w,h=h,start=np.array([x+(w-1)/2,y+(h-1)/2],float)))
# Component order: large triangle, large square, small square,
# small triangle, medium square, medium triangle.
group_centers={1:(160,430),2:(330,430),4:(460,430),0:(640,610),3:(780,610),5:(900,610)}
order=[2,4,1,3,5,0]
left=81
for i in order:
    s=shapes[i]
    s['end']=np.array([left+(s['w']-1)/2,512.])
    left+=s['w']+30+(30 if i==1 else 0)
for i,s in enumerate(shapes): s['group']=np.array(group_centers[i],float)

def smooth(t):
    t=np.clip(t,0,1)
    return t*t*t*(10+t*(-15+6*t))

proc=subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','bgr24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','16','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')],stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
for frame in range(96):
    canvas=np.empty_like(original); canvas[:]=bg
    for i,s in enumerate(shapes):
        if frame<=43:
            u=smooth(frame/43)
            p=s['start']*(1-u)+s['group']*u
        else:
            u=smooth((frame-43)/48)
            p=s['group']*(1-u)+s['end']*u
            # Separate crossing paths while sorting within each type.
            arc={1:-160,2:115,4:0,0:160,3:-100,5:0}[i]
            p[1]+=arc*np.sin(np.pi*u)
        x=int(round(p[0]-(s['w']-1)/2)); y=int(round(p[1]-(s['h']-1)/2))
        patch=canvas[y:y+s['h'],x:x+s['w']]
        patch[s['mask']]=s['pixels'][s['mask']]
    if frame==0:
        assert np.array_equal(canvas,original)
    proc.stdin.write(canvas.tobytes())
proc.stdin.close()
assert proc.wait()==0
