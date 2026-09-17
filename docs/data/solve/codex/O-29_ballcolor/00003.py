from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path('/app')
BG=(240,240,240)
source=Image.open(ROOT/'first_frame.png').convert('RGB')
red=(255,50,50)
# Original ball sprites retain the source's outline and rasterization.
def sprite(x,y):
    return source.crop((x-22,y-22,x+23,y+23))
sprites={'A':sprite(323,463),'B':sprite(331,818),'C':sprite(189,199)}
B=[(331,818),(381,818),(331,868),(381,868)]
C=[(189+50*c,199+50*r) for r in range(3) for c in range(3)]+[(189,349)]
Bt=[(423,613),(323,663),(373,663),(423,663)]
Ct=[(473+50*c,463+50*r) for r in range(5) for c in range(2)]
base=source.copy()
d=ImageDraw.Draw(base)
for x,y in B+C:
    d.rectangle((x-22,y-22,x+22,y+22),fill=BG)
for box in [(352,270,496,314),(414,840,526,882),(486,535,633,576)]:
    d.rectangle(box,fill=BG)
labels={'A':source.crop((486,535,634,577)), 'B':source.crop((414,840,527,883)), 'C':source.crop((352,270,497,315))}
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',50)
def label(im,n,x):
    if n==11:
        im.paste(labels['A'],(x,535)); return
    draw=ImageDraw.Draw(im)
    text=f'A: {n}'
    bounds=draw.textbbox((x+3,527),text,font=font)
    draw.rectangle((x,535,bounds[2]+3,576),fill='white')
    draw.text((x+3,527),text,font=font,fill='black')
def ease(t):
    t=max(0,min(1,t)); return t*t*(3-2*t)
def balls(im,starts,ends,t,kind):
    u=ease(t)
    for (x,y),(tx,ty) in zip(starts,ends):
        px=round(x+(tx-x)*u); py=round(y+(ty-y)*u)
        im.paste(sprites['A' if t>=1 else kind],(px-22,py-22))
def frame(i):
    if i==0: return source.copy()
    im=base.copy()
    tb=(i-6)/18
    tc=(i-32)/22
    balls(im,B,Bt,tb,'B')
    balls(im,C,Ct,tc,'C')
    if i<=6: im.paste(labels['B'],(414,840))
    if i<=32: im.paste(labels['C'],(352,270))
    # The label moves clear of the new columns before they arrive.
    lx=486+round(100*ease((i-27)/5))
    label(im,25 if tc>=1 else 15 if tb>=1 else 11,lx)
    return im

out=ROOT/'output'; out.mkdir(exist_ok=True)
proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','16','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')],stdin=subprocess.PIPE)
for i in range(61):
    proc.stdin.write(np.asarray(frame(i)).tobytes())
proc.stdin.close()
if proc.wait(): raise RuntimeError('ffmpeg encoding failed')
