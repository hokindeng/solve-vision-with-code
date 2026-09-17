from PIL import Image, ImageDraw
import numpy as np
import cv2
import os

ROOT='/app'
base=Image.open(ROOT+'/first_frame.png').convert('RGB')
BG=(245,245,245); PURPLE=(138,43,226); EDGE=(50,100,140)
RED=(210,53,59); GREEN=(34,157,79); BLUE=(45,125,218)
boxes=[(50,829,235,1014),(296,829,481,1014),(542,829,727,1014),(788,829,973,1014)]
shapes=[[(0,0),(1,0),(1,1),(2,1)],[(0,0),(1,0)],[(0,0),(1,0),(1,1)],[(0,0),(1,0),(0,1)]]

def piece(im, cells, x, y, pitch, opacity=255):
    layer=Image.new('RGBA', im.size)
    d=ImageDraw.Draw(layer)
    for cx,cy in cells:
        xx=round(x+cx*pitch); yy=round(y+cy*pitch)
        d.rectangle((xx,yy,xx+round(pitch)-2,yy+round(pitch)-2),fill=(*PURPLE,opacity),outline=(*EDGE,opacity))
    im.paste(Image.alpha_composite(im.convert('RGBA'),layer).convert('RGB'))

def status(im,i,correct):
    x0,y0,x1,y1=boxes[i]
    color=GREEN if correct else RED
    # Recolor only the existing neutral panel interior, retaining its piece and number.
    a=np.asarray(im).copy()
    region=a[y0+1:y1,x0+1:x1]
    mask=np.all(region==(230,230,230),axis=2)
    region[mask]=(221,241,226) if correct else (247,222,223)
    im.paste(Image.fromarray(a))
    d=ImageDraw.Draw(im)
    d.rectangle(boxes[i],outline=color,width=3)
    x=x1-28; y=y0+24
    if correct:
        d.line([(x-10,y),(x-3,y+8),(x+12,y-11)],fill=color,width=5)
    else:
        d.line([(x-9,y-9),(x+9,y+9)],fill=color,width=4)
        d.line([(x-9,y+9),(x+9,y-9)],fill=color,width=4)

def clear_gap(im):
    d=ImageDraw.Draw(im)
    d.rectangle((486,383,589,434),fill=BG)
    d.rectangle((486,435,537,486),fill=BG)

os.makedirs(ROOT+'/output',exist_ok=True)
raw=ROOT+'/output/render.mp4'
writer=cv2.VideoWriter(raw,cv2.VideoWriter_fourcc(*'mp4v'),16,(1024,1024))
for f in range(99):
    im=base.copy()
    # Four eighteen-frame examinations, with a preview followed by a verdict.
    for i in range(4):
        start=5+i*18
        age=f-start
        if age>=12:
            status(im,i,i==3)
        elif age>=0:
            ImageDraw.Draw(im).rectangle(boxes[i],outline=BLUE,width=4)
        if 4<=age<12:
            # A translucent overlay shows both uncovered cells and any overlap.
            piece(im,shapes[i],486,383,52,155)
    if f>=77:
        # Lift the accepted piece from its original panel, then enlarge it to fit.
        ImageDraw.Draw(im).rectangle((842,883,916,957),fill=(221,241,226))
        t=min(1,(f-77)/17)
        t=t*t*(3-2*t)
        if t==1:
            clear_gap(im)
        piece(im,shapes[3],842+(486-842)*t,883+(383-883)*t,38+14*t)
    writer.write(cv2.cvtColor(np.array(im),cv2.COLOR_RGB2BGR))
writer.release()
import subprocess
subprocess.run(['ffmpeg','-y','-loglevel','error','-i',raw,'-c:v','libx264','-crf','12','-pix_fmt','yuv420p','-r','16',ROOT+'/output/video.mp4'],check=True)
os.remove(raw)
