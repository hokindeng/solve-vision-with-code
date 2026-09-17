from PIL import Image, ImageDraw
import numpy as np
import cv2
import subprocess
from pathlib import Path

ROOT=Path('/app')
BASE=Image.open(ROOT/'first_frame.png').convert('RGB')
A=np.array(BASE)
BOXES=[(50,829,235,1014),(296,829,481,1014),(542,829,727,1014),(788,829,973,1014)]
SHAPES=[[(1,0),(2,0),(0,1),(1,1),(0,2)],[(0,0),(1,0),(1,1)],[(0,0),(0,1),(1,1),(0,2)],[(1,0),(0,1),(1,1),(1,2)]]
STARTS=[4,22,40,58]
RED=(205,55,55); GREEN=(35,158,75); BLUE=(40,126,220)

def box(im,index,state):
    x0,y0,x1,y1=BOXES[index]
    tint={'active':(225,237,250),'bad':(250,218,218),'good':(218,242,222)}[state]
    region=np.array(im)
    m=np.all(A[y0+1:y1,x0+1:x1]==(230,230,230),axis=2)
    patch=region[y0+1:y1,x0+1:x1]
    patch[m]=tint
    im=Image.fromarray(region)
    d=ImageDraw.Draw(im)
    color={'active':BLUE,'bad':RED,'good':GREEN}[state]
    d.rectangle((x0,y0,x1,y1),outline=color,width=3)
    if state=='bad':
        d.line([(x1-32,y0+12),(x1-13,y0+31)],fill=color,width=4)
        d.line([(x1-13,y0+12),(x1-32,y0+31)],fill=color,width=4)
    elif state=='good':
        d.line([(x1-38,y0+23),(x1-28,y0+33),(x1-12,y0+12)],fill=color,width=4)
    return im

def piece(d,shape,x,y,size,fill,edge):
    for cx,cy in shape:
        l=round(x+cx*size); t=round(y+cy*size)
        r=round(x+(cx+1)*size)-2; b=round(y+(cy+1)*size)-2
        d.rectangle((l,t,r,b),fill=fill,outline=edge,width=1)

def frame(n):
    im=BASE.copy()
    if n==0: return im
    for i,start in enumerate(STARTS):
        if n>=start:
            im=box(im,i,'active' if n<start+12 else ('good' if i==3 else 'bad'))
    for i,start in enumerate(STARTS):
        if start+4<=n<start+16:
            # Transparent previews preserve the visible gap and existing blocks.
            overlay=Image.new('RGBA',im.size)
            d=ImageDraw.Draw(overlay)
            color=BLUE if n<start+12 else (GREEN if i==3 else RED)
            px=434 if i==2 else 382
            piece(d,SHAPES[i],px,228,52,(*color,95),(*color,255))
            im=Image.alpha_composite(im.convert('RGBA'),overlay).convert('RGB')
    if n>=76:
        # Lift the correct piece from its candidate box and enlarge it to the grid.
        d=ImageDraw.Draw(im)
        for cx,cy in SHAPES[3]:
            d.rectangle((842+38*cx,864+38*cy,878+38*cx,900+38*cy),fill=(218,242,222))
        t=min(1,(n-76)/17)
        s=t*t*(3-2*t)
        if n>=93:
            arr=np.array(im)
            arr[np.all(A==(255,100,100),axis=2)]=(245,245,245)
            im=Image.fromarray(arr)
        d=ImageDraw.Draw(im)
        piece(d,SHAPES[3],842+(382-842)*s,864+(228-864)*s,38+14*s,(200,100,0),(50,100,140))
    return im

def main():
    out=ROOT/'output'; out.mkdir(exist_ok=True)
    cmd=['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-preset','slow','-crf','0','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    for n in range(99): proc.stdin.write(np.array(frame(n)).tobytes())
    proc.stdin.close()
    err=proc.stderr.read(); rc=proc.wait()
    if rc: raise RuntimeError(err.decode())
if __name__=='__main__': main()
