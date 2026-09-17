from PIL import Image, ImageDraw
import numpy as np
import cv2
import os

ROOT='/app'
BASE=Image.open(ROOT+'/first_frame.png').convert('RGB')
BOXES=[(50,829,235,1014),(296,829,481,1014),(542,829,727,1014),(788,829,973,1014)]
SHAPES=[[(0,0),(0,1)],[(0,0),(1,0),(2,0)],[(0,0)],[(0,0),(1,0)]]
RED=(213,65,65); GREEN=(37,157,82); BLUE=(48,133,211)

def frame_box(im,i,color,verdict=False):
    d=ImageDraw.Draw(im)
    x,y,r,b=BOXES[i]
    d.rectangle((x,y,r,b),outline=color,width=4)
    if verdict:
        cx=r-23; cy=y+23
        if i==3:
            d.line([(cx-10,cy),(cx-3,cy+8),(cx+12,cy-10)],fill=color,width=5)
        else:
            d.line((cx-9,cy-9,cx+9,cy+9),fill=color,width=5)
            d.line((cx-9,cy+9,cx+9,cy-9),fill=color,width=5)

def tiles(im,shape,x,y,size=52,ghost=False,color=BLUE):
    if ghost:
        layer=Image.new('RGBA',im.size,(0,0,0,0)); d=ImageDraw.Draw(layer)
        for a,b in shape:
            xx=round(x+a*size); yy=round(y+b*size)
            d.rectangle((xx,yy,round(xx+size-1),round(yy+size-1)),fill=(*color,75),outline=(*color,255),width=2)
        im.paste(Image.alpha_composite(im.convert('RGBA'),layer).convert('RGB'))
    else:
        d=ImageDraw.Draw(im)
        for a,b in shape:
            xx=round(x+a*size); yy=round(y+b*size)
            d.rectangle((xx,yy,round(x+(a+1)*size)-1,round(y+(b+1)*size)-1),fill=(105,105,105),outline=(50,100,140),width=1)

def render(f):
    im=BASE.copy()
    if f<5:return im
    # Each inspection has a selection, an overlaid fit preview, and a verdict.
    for i in range(4):
        start=5+17*i; local=f-start
        if local<0:break
        if local>=12:
            frame_box(im,i,GREEN if i==3 else RED,True)
        else:
            frame_box(im,i,BLUE)
        if 4<=local<16 and f<73:
            color=BLUE if local<12 else (GREEN if i==3 else RED)
            tiles(im,SHAPES[i],486,383,ghost=True,color=color)
    if f>=73:
        # Lift the winning piece from its card and enlarge it to blueprint scale.
        d=ImageDraw.Draw(im)
        d.rectangle((842,902,916,938),fill=(230,230,230))
        t=min(1,(f-73)/19)
        ease=t*t*(3-2*t)
        x=842+(486-842)*ease
        y=902+(383-902)*ease
        size=37.5+(52-37.5)*ease
        if t==1:
            d.rectangle((486,383,589,434),fill=(245,245,245))
        tiles(im,SHAPES[3],x,y,size)
    return im

def main():
    os.makedirs(ROOT+'/output',exist_ok=True)
    # Feed original RGB frames directly to ffmpeg for standards-compliant output.
    import subprocess
    p=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','15','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',ROOT+'/output/video.mp4'],stdin=subprocess.PIPE)
    for f in range(99):p.stdin.write(np.asarray(render(f)).tobytes())
    p.stdin.close()
    if p.wait():raise RuntimeError('ffmpeg failed')

if __name__=='__main__': main()
