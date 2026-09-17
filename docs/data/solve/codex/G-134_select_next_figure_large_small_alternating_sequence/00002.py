from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os

BASE='/app'
os.makedirs(BASE+'/output', exist_ok=True)
source=Image.open(BASE+'/first_frame.png').convert('RGB')
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
red=(220,35,45)
teal=source.getpixel((230,330))
# All drawings are confined to the sequence explanation and selected answer.
def frame(i):
    im=source.copy()
    d=ImageDraw.Draw(im)
    # Read each member in order; temporary outlines track the size pattern.
    steps=[(5,14,(211,307,271,367),'SMALL'),
           (15,24,(376,292,466,382),'LARGE'),
           (25,34,(572,307,632,367),'SMALL')]
    for start,end,box,label in steps:
        if start<=i<=end:
            x0,y0,x1,y1=box
            d.rectangle((x0-8,y0-8,x1+8,y1+8),outline=red,width=3)
            d.text(((x0+x1)/2,410),label,font=font,fill=red,anchor='mt')
    # Resolve the missing member as a large square, revealing it from its center.
    if i>=35:
        d.rectangle((772,324,792,347),fill='white')
        progress=min(1,(i-34)/8)
        side=round(91*progress)
        left=782-side//2
        top=337-side//2
        d.rectangle((left,top,left+side-1,top+side-1),fill=teal)
    if 35<=i<=44:
        d.text((782,410),'LARGE',font=font,fill=red,anchor='mt')
    # Trace a red circle around the matching first option, then hold the result.
    if i>=44:
        progress=min(1,(i-43)/11)
        d.arc((89,777,243,931),start=-90,end=-90+360*progress,fill=red,width=5)
    return im

path=BASE+'/output/video.mp4'
# Pipe RGB frames directly to ffmpeg to produce the required H.264 format.
import subprocess
proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',path],stdin=subprocess.PIPE)
for i in range(60):
    proc.stdin.write(np.asarray(frame(i)).tobytes())
proc.stdin.close()
if proc.wait()!=0:
    raise RuntimeError('ffmpeg failed')
