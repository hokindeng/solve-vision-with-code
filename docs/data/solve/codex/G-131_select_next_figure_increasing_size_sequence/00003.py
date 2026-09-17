from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import math
import subprocess
from pathlib import Path

ROOT=Path('/app')
OUT=ROOT/'output'
OUT.mkdir(exist_ok=True)
base=Image.open(ROOT/'first_frame.png').convert('RGB')
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',21)
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
ink=(65,65,65)
teal=(0,177,179)

def label(d,xy,text,f=font,fill=ink):
    d.text(xy,text,font=f,fill=fill,anchor='mm')

def measurement(d,x,w):
    y=395
    a=x-(w-1)/2; b=x+(w-1)/2
    d.line((a,y,b,y),fill=ink,width=2)
    for xx in (a,b): d.line((xx,y-5,xx,y+5),fill=ink,width=2)
    label(d,(x,420),f'{w} px',small)

def step(d,a,b):
    y=461
    d.line((a,y,b,y),fill=ink,width=2)
    d.line((b-7,y-5,b,y,b-7,y+5),fill=ink,width=2)
    label(d,((a+b)/2,y-19),'+18 px',small)

frames=[]
for i in range(60):
    im=base.copy(); d=ImageDraw.Draw(im)
    if i>=5: measurement(d,241,31)
    if i>=12:
        measurement(d,421,49)
        step(d,272,390)
    if i>=21:
        measurement(d,602,67)
        step(d,453,570)
    if i>=30:
        step(d,637,747)
        measurement(d,782,85)
        label(d,(512,528),'31 → 49 → 67 → 85 pixels')
    if i>=37:
        # The missing member uses the exact pixels of the matching choice.
        patch=base.crop((814,815,899,914))
        arr=np.asarray(patch)
        mask=Image.fromarray(((arr[:,:,0]<10)&(arr[:,:,1]>130)&(arr[:,:,2]>130)).astype('uint8')*255)
        im.paste(patch,(740,288),mask)
        d=ImageDraw.Draw(im)
    if i>=43:
        # Draw the requested circle progressively over the final second.
        progress=min(1,(i-42)/13)
        points=[]
        for t in np.linspace(-math.pi/2,-math.pi/2+2*math.pi*progress, max(2,int(300*progress))):
            points.append((856+77*math.cos(t),864+77*math.sin(t)))
        d.line(points,fill=(230,25,35),width=5)
    frames.append(np.array(im))
# Lossless source frames feed H.264 at the requested frame rate/pixel format.
p=subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-preset','slow','-crf','12','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')],stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
for frame in frames: p.stdin.write(frame.tobytes())
p.stdin.close()
if p.wait()!=0: raise RuntimeError('ffmpeg failed')
assert np.array_equal(frames[0],np.asarray(base))
Image.fromarray(frames[-1]).save(OUT/'last_frame.png')
print(OUT/'video.mp4')
