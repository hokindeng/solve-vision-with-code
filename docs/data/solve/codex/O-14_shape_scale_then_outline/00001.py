from PIL import Image, ImageDraw
import numpy as np
import cv2
import os

BASE='/app'
source=Image.open(BASE+'/first_frame.png').convert('RGB')
os.makedirs(BASE+'/output',exist_ok=True)
# The example scales both dimensions by about 1.23, then strengthens
# the green outline. Apply those same steps in the two answer positions.
frames=[np.array(source)]
green=(0,100,0)

def rectangle(draw,cx,cy,w,h,stroke):
    draw.rectangle((round(cx-w/2),round(cy-h/2),round(cx+w/2),round(cy+h/2)),outline=green,width=stroke)

for i in range(1,16):
    frame=source.copy()
    draw=ImageDraw.Draw(frame)
    # Replace only the first answer area during the size transformation.
    draw.rectangle((393,643,567,721),fill='white')
    t=min(i/8,1)
    smooth=t*t*(3-2*t)
    rectangle(draw,480,682,133+30*smooth,32+8*smooth,1)
    if i>=9:
        draw.rectangle((699,643,875,721),fill='white')
        u=(i-8)/7
        # Expand the border about its original center line as in the example.
        stroke=round(1+5*u)
        growth=stroke-1
        rectangle(draw,787,682,163+growth,40+growth,stroke)
    frames.append(np.array(frame))

# Encode losslessly in H.264, with the requested 4:2:0 pixel format.
import subprocess
proc=subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo',
    '-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-',
    '-an','-c:v','libx264','-qp','0','-pix_fmt','yuv420p',
    '-movflags','+faststart',BASE+'/output/video.mp4'],stdin=subprocess.PIPE,
    stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
_,err=proc.communicate(b''.join(x.tobytes() for x in frames))
if proc.returncode:
    raise RuntimeError(err.decode())
