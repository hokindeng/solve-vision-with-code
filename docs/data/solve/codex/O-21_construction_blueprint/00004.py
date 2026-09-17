from PIL import Image, ImageDraw
import numpy as np
import subprocess, os

ROOT='/app'
BASE=Image.open(ROOT+'/first_frame.png').convert('RGB')
BLUE=(40,80,150)
RED=(211,56,62)
GREEN=(36,153,77)
BOXES=[(50,829,235,1014),(296,829,481,1014),(542,829,727,1014),(788,829,973,1014)]
SHAPES=[[(0,1),(1,1),(2,1)],[(0,0),(1,0),(2,0),(1,1)],[(1,1),(2,0),(2,1)],[(1,0),(0,1),(1,1),(2,1)]]

def verdict(im,k):
    x0,y0,x1,y1=BOXES[k]
    color=GREEN if k==3 else RED
    a=np.array(im)
    region=a[y0+1:y1,x0+1:x1]
    mask=np.all(region==[230,230,230],axis=2)
    region[mask]=(225,242,230) if k==3 else (249,225,225)
    im=Image.fromarray(a)
    d=ImageDraw.Draw(im)
    d.rectangle((x0,y0,x1,y1),outline=color,width=3)
    cx=x1-25;cy=y0+23
    if k==3:
        d.line([(cx-10,cy),(cx-3,cy+8),(cx+12,cy-10)],fill=color,width=4)
    else:
        d.line((cx-8,cy-8,cx+8,cy+8),fill=color,width=4)
        d.line((cx-8,cy+8,cx+8,cy-8),fill=color,width=4)
    return im

def piece(im,shape,x,y,size,opacity=255,outline=None):
    layer=Image.new('RGBA',im.size)
    d=ImageDraw.Draw(layer)
    for col,row in shape:
        r=(round(x+col*size),round(y+row*size),round(x+(col+1)*size)-2,round(y+(row+1)*size)-2)
        d.rectangle(r,fill=(*BLUE,opacity),outline=(*outline,255) if outline else None,width=2)
    return Image.alpha_composite(im.convert('RGBA'),layer).convert('RGB')

def frame(n):
    im=BASE.copy()
    if n==0:return im
    for k in range(4):
        start=3+k*18
        if n>=start+13: im=verdict(im,k)
    if n<75:
        for k in range(4):
            start=3+k*18
            if start<=n<start+18:
                if n<start+13:
                    ImageDraw.Draw(im).rectangle(BOXES[k],outline=(225,167,38),width=4)
                if start+5<=n<start+17:
                    im=piece(im,SHAPES[k],330,383,52,145,GREEN if k==3 and n>=start+13 else RED if n>=start+13 else (225,167,38))
    else:
        # The selected piece leaves its candidate box and grows to the blueprint scale.
        d=ImageDraw.Draw(im)
        d.rectangle((860,882,899,920),fill=(225,242,230))
        d.rectangle((822,920,936,958),fill=(225,242,230))
        t=min(1,(n-75)/18)
        t=t*t*(3-2*t)
        x=823+(330-823)*t
        y=883+(383-883)*t
        size=38+14*t
        if t==1:
            a=np.array(im)
            original=np.array(BASE)
            mask=np.all(original==[255,100,100],axis=2)
            a[mask]=[245,245,245]
            im=Image.fromarray(a)
        im=piece(im,SHAPES[3],x,y,size)
    return im

os.makedirs(ROOT+'/output',exist_ok=True)
p=subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','16','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',ROOT+'/output/video.mp4'],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
for n in range(99):p.stdin.write(frame(n).tobytes())
p.stdin.close()
err=p.stderr.read();p.wait()
if p.returncode:raise RuntimeError(err.decode())
