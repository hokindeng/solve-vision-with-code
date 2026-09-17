from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path('/app')
OUT=ROOT/'output'
OUT.mkdir(exist_ok=True)
base=Image.open(ROOT/'first_frame.png').convert('RGB')
a=np.array(base)
colors=[(255,0,0),(255,200,0),(0,200,0),(255,200,0)]
# Yellow at North is the Red -> Yellow -> Green phase.
lights=[((255,0,0),220,630,0),((255,200,0),512,338,7),((0,200,0),804,630,11)]
masks=[np.all(a==rgb,axis=2) for rgb,_,_,_ in lights]
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',100)
# Reuse the supplied typography exactly for digits already in the image.
digits={1:base.crop((451,276,574,400)),4:base.crop((159,568,282,692))}
for n in (2,3):
    tile=Image.new('RGB',(123,124),'white')
    d=ImageDraw.Draw(tile)
    d.text((61-font.getlength(str(n))/2,4),str(n),font=font,fill='black')
    digits[n]=tile

cmd=['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','0','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')]
proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
for frame in range(112):
    elapsed=min(frame//16,6)
    if elapsed==0:
        im=base
    else:
        arr=a.copy()
        states=[]
        for mask,(_,cx,cy,offset) in zip(masks,lights):
            phase=(offset+elapsed)%16
            arr[mask]=colors[phase//4]
            states.append((cx,cy,4-phase%4))
        im=Image.fromarray(arr)
        for cx,cy,count in states:
            im.paste(digits[count],(cx-61,cy-62))
    proc.stdin.write(im.tobytes())
proc.stdin.close()
err=proc.stderr.read()
if proc.wait():
    raise RuntimeError(err.decode())
