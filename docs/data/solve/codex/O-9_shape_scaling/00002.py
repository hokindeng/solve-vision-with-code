from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT=Path('/app')
OUT=ROOT/'output'
OUT.mkdir(exist_ok=True)
base=Image.open(ROOT/'first_frame.png').convert('RGB')
# Hexagon radii: 75 px in A and 65 px in B.
scale=65/75
box=(700,680,839,857)
original=base.crop(box)
blank=Image.new('RGB',original.size,'white')
def ease(t):
    t=max(0,min(1,t))
    return t*t*(3-2*t)

proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-preset','slow','-crf','12','-pix_fmt','yuv420p',str(OUT/'video.mp4')],stdin=subprocess.PIPE)
for i in range(60):
    frame=base.copy()
    if i>5:
        # Remove the placeholder, introduce C, and smoothly shrink it
        # by precisely the same factor as the example.
        fade=ease((i-5)/12)
        patch=Image.blend(original,blank,fade)
        if i>=17:
            patch=blank.copy()
            shrink=ease((i-23)/30)
            s=1+(scale-1)*shrink
            shape=blank.copy()
            d=ImageDraw.Draw(shape)
            cx,cy=769-box[0],768-box[1]
            pts=[(cx,cy-75*s),(cx+52*s,cy),(cx,cy+75*s),(cx-52*s,cy)]
            d.polygon(pts, fill=(255,128,0),outline=(0,0,0),width=1)
            patch=Image.blend(blank,shape,ease((i-17)/9))
        frame.paste(patch,box[:2])
    proc.stdin.write(np.asarray(frame).tobytes())
proc.stdin.close()
if proc.wait():
    raise RuntimeError('ffmpeg failed')
