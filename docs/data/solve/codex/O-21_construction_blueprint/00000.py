from PIL import Image, ImageDraw
import numpy as np
import subprocess, os

ROOT='/app'
base=Image.open(ROOT+'/first_frame.png').convert('RGB')
BG=(245,245,245); BLUE=(70,130,180); EDGE=(50,100,140)
RED=(211,61,66); GREEN=(43,156,82); ACTIVE=(235,168,43)
boxes=[(50,829,235,1014),(296,829,481,1014),(542,829,727,1014),(788,829,973,1014)]
shapes=[[(0,0),(1,0),(1,1),(2,1),(3,1),(2,2)],[(2,0),(0,1),(1,1),(2,1),(2,2)],[(0,0),(0,1),(0,2),(1,2)],[(1,0),(1,1),(0,2),(1,2),(2,2)]]
gap=[(434,331),(434,383),(382,435),(434,435),(486,435)]

def tile(draw,x,y,size,fill=BLUE):
    x,y=round(x),round(y); size=round(size)
    draw.rectangle((x,y,x+size-1,y+size-1),fill=fill,outline=EDGE,width=1)

def status(im,i,good):
    d=ImageDraw.Draw(im); x0,y0,x1,y1=boxes[i]; c=GREEN if good else RED
    # Tint the box interior while preserving the original blue candidate and label.
    ar=np.array(im); orig=np.array(base)
    crop=ar[y0+1:y1,x0+1:x1]; mask=np.all(orig[y0+1:y1,x0+1:x1]==(230,230,230),axis=2)
    crop[mask]=(224,240,227) if good else (247,225,225)
    im.paste(Image.fromarray(ar)); d=ImageDraw.Draw(im)
    d.rectangle((x0,y0,x1,y1),outline=c,width=3)
    if good:
        d.line([(x1-42,y0+25),(x1-32,y0+36),(x1-15,y0+13)],fill=c,width=5)
    else:
        d.line([(x1-36,y0+15),(x1-16,y0+35)],fill=c,width=5)
        d.line([(x1-16,y0+15),(x1-36,y0+35)],fill=c,width=5)

def preview(im,i,alpha):
    layer=im.copy(); d=ImageDraw.Draw(layer)
    ox=434 if i==2 else 382
    for x,y in shapes[i]:
        tile(d,ox+x*52,331+y*52,51)
        d.rectangle((ox+x*52,331+y*52,ox+x*52+50,331+y*52+50),outline=GREEN if i==3 else ACTIVE,width=2)
    return Image.blend(im,layer,alpha)

def frame(n):
    im=base.copy()
    # Four 16-frame examinations: select, preview, verdict, and clear.
    for i in range(4):
        start=5+16*i
        if n>=start+12: status(im,i,i==3)
        elif n>=start:
            d=ImageDraw.Draw(im); d.rectangle(boxes[i],outline=ACTIVE,width=4)
        if start+3<=n<start+14:
            opacity=min(0.72,(n-start-2)*0.22)
            im=preview(im,i,opacity)
    if n>=71:
        d=ImageDraw.Draw(im)
        # The selected piece leaves its box and grows to the structure's cell size.
        for x,y in shapes[3]:
            d.rectangle((838+x*28,879+y*28,838+x*28+26,879+y*28+26),fill=(224,240,227))
        t=min(1,(n-71)/21); t=t*t*(3-2*t)
        if t>=1:
            for x,y in gap: d.rectangle((x,y,x+51,y+51),fill=BG)
        ox=838+(382-838)*t; oy=879+(331-879)*t
        step=28+24*t; size=27+24*t
        for x,y in shapes[3]: tile(d,ox+x*step,oy+y*step,size)
    return im

os.makedirs(ROOT+'/output',exist_ok=True)
p=subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',ROOT+'/output/video.mp4'],stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
for n in range(99): p.stdin.write(np.asarray(frame(n)).tobytes())
p.stdin.close()
if p.wait(): raise RuntimeError('ffmpeg failed')
