from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os

BASE='/app'
os.makedirs(BASE+'/output',exist_ok=True)
original=Image.open(BASE+'/first_frame.png').convert('RGB')
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
red=(222,38,44)

def frame(n):
    im=original.copy()
    # Annotations build from left to right, following the alternating sizes.
    layer=Image.new('RGBA',(1024,1024))
    d=ImageDraw.Draw(layer)
    steps=[(5,241,337,28,'SMALL'),(15,421,337,52,'LARGE'),(25,602,337,28,'SMALL')]
    for start,x,y,r,label in steps:
        if n>=start:
            p=min(1,(n-start+1)/6)
            d.arc((x-r,y-r,x+r,y+r),-90,-90+360*p,fill=red+(255,),width=3)
            if p==1:
                d.text((x,412),label,font=font,fill=red+(255,),anchor='mt')
    if n>=35:
        alpha=int(255*min(1,(n-34)/6))
        d.text((782,412),'LARGE',font=font,fill=red+(alpha,),anchor='mt')
    if n>=43:
        p=min(1,(n-42)/12)
        d.arc((102,790,230,918),-90,-90+360*p,fill=red+(255,),width=5)
    return Image.alpha_composite(im.convert('RGBA'),layer).convert('RGB')

# Feed lossless source frames to ffmpeg for the requested playback format.
import subprocess
proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','16','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',BASE+'/output/video.mp4'],stdin=subprocess.PIPE)
for n in range(60):
    proc.stdin.write(np.asarray(frame(n)).tobytes())
proc.stdin.close()
if proc.wait()!=0:
    raise RuntimeError('ffmpeg failed')
