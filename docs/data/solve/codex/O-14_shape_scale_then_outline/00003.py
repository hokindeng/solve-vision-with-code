from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT=Path('/app')
OUT=ROOT/'output'
OUT.mkdir(exist_ok=True)
base=Image.open(ROOT/'first_frame.png').convert('RGB')
color=(175,191,95)
# Work only in the two answer cells. All surrounding artwork is copied.
cells=[(435,622,526,742),(743,622,834,742)]

def smooth(t):
    t=max(0,min(1,t))
    return t*t*(3-2*t)

def answer(cell, side, width):
    x0,y0,x1,y1=cell
    s=4
    im=Image.new('RGB',((x1-x0)*s,(y1-y0)*s),'white')
    d=ImageDraw.Draw(im)
    cx=(480 if x0==435 else 787)-x0
    cy=682-y0
    # Nested rectangles retain the white center of the source shape.
    d.rectangle([(cx-side/2)*s,(cy-side/2)*s,(cx+side/2)*s,(cy+side/2)*s],fill=color)
    d.rectangle([(cx-side/2+width)*s,(cy-side/2+width)*s,(cx+side/2-width)*s,(cy+side/2-width)*s],fill='white')
    return im.resize((x1-x0,y1-y0),Image.Resampling.LANCZOS)

frames=[]
for i in range(16):
    frame=base.copy()
    if i:
        p=smooth(i/8)
        # The middle answer appears while scaling to 80 percent.
        side=104*(1-.2*p)
        # Larger early shapes must fit their answer cell, so reveal after shrinking begins.
        cell=(420,622,541,742)
        # Render using the actual center in this expanded local patch.
        s=4
        im=Image.new('RGB',(121*s,120*s),'white')
        d=ImageDraw.Draw(im)
        cx,cy=60,60
        w=6*(1-.2*p)
        d.rectangle(((cx-side/2)*s,(cy-side/2)*s,(cx+side/2)*s,(cy+side/2)*s),fill=color)
        d.rectangle(((cx-side/2+w)*s,(cy-side/2+w)*s,(cx+side/2-w)*s,(cy+side/2-w)*s),fill='white')
        im=im.resize((121,120),Image.Resampling.LANCZOS)
        frame.paste(Image.blend(base.crop(cell),im,smooth(i/4)),cell[:2])
        if i>8:
            q=smooth((i-8)/7)
            cell=cells[1]
            im=answer(cell,83.2,4.8*(1-q)+1.2*q)
            frame.paste(Image.blend(base.crop(cell),im,smooth((i-8)/3)),cell[:2])
    frames.append(np.asarray(frame))
proc=subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','12','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
_,err=proc.communicate(b''.join(f.tobytes() for f in frames))
if proc.returncode: raise RuntimeError(err.decode())
