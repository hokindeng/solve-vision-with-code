from PIL import Image, ImageDraw
import numpy as np
import cv2
import os

ROOT='/app'
base=Image.open(ROOT+'/first_frame.png').convert('RGB')
os.makedirs(ROOT+'/output', exist_ok=True)

def smooth(t):
    t=max(0,min(1,t))
    return t*t*(3-2*t)

def rectangle(cx,cy,w,h,stroke):
    # Supersampling keeps the diagram's clean, softly antialiased edges.
    s=4
    layer=Image.new('RGBA',(1024*s,1024*s))
    d=ImageDraw.Draw(layer)
    d.rectangle((round((cx-w/2)*s),round((cy-h/2)*s),round((cx+w/2)*s),round((cy+h/2)*s)),outline=(119,48,157,255),width=round(stroke*s))
    return layer.resize((1024,1024),Image.Resampling.LANCZOS)

def stage(im,box,progress,cx,mode):
    if progress<=0:return im
    # Only the answer's question mark and replacement shape may change.
    a=smooth(progress/0.23)
    patch=base.crop(box)
    im.paste(Image.blend(patch,Image.new('RGB',patch.size,'white'),a),box)
    if mode==1:
        q=smooth(progress)
        w=194+(136-194)*q
        h=50+(35-50)*q
        stroke=6
    else:
        w,h=136,35
        stroke=6-3*smooth(progress)
    layer=rectangle(cx,683,w,h,stroke)
    layer.putalpha(layer.getchannel('A').point(lambda x:round(x*a)))
    return Image.alpha_composite(im.convert('RGBA'),layer).convert('RGB')

frames=[]
for i in range(16):
    im=base.copy()
    if i:
        im=stage(im,(446,644,515,720),min(i/8,1),480,1)
        im=stage(im,(751,644,821,720),max(0,(i-8)/7),787,2)
    frames.append(np.asarray(im))
# Lossless H.264 minimizes codec changes to the otherwise fixed source pixels.
import subprocess
p=subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','0','-preset','medium','-pix_fmt','yuv420p',ROOT+'/output/video.mp4'],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
_,err=p.communicate(b''.join(f.tobytes() for f in frames))
if p.returncode:raise RuntimeError(err.decode())
