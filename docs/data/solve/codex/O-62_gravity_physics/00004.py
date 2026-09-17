from PIL import Image, ImageDraw, ImageFont
import numpy as np
import math
import subprocess
from pathlib import Path

ROOT=Path('/app')
OUT=ROOT/'output/video.mp4'
OUT.parent.mkdir(exist_ok=True)
source=Image.open(ROOT/'first_frame.png').convert('RGB')
background=source.copy()
ImageDraw.Draw(background).rectangle((530,245,766,351), fill='white')
sprite=source.crop((691,277,765,351)).convert('RGBA')
a=np.array(sprite)
a[:,:,3]=np.where(np.all(a[:,:,:3]==255,axis=2),0,255)
sprite=Image.fromarray(a)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',30*3)
g=6.1
elasticity=.70
h0=13.3
v0=.7
impact=math.sqrt(v0*v0+2*g*h0)
tfirst=(v0+impact)/g
# Analytic ballistic arcs, with instantaneous restitution at each contact.
def state(t):
    if t<tfirst:
        return h0+v0*t-.5*g*t*t,v0-g*t
    t-=tfirst
    u=impact*elasticity
    while u*u/(2*g)>.004:
        duration=2*u/g
        if t<duration:
            return max(0,u*t-.5*g*t*t),u-g*t
        t-=duration
        u*=elasticity
    return 0.,0.

def frame(i):
    if i==0:
        return source
    h,v=state(i/16)
    bottom=774-h*(425/13.3)
    top=round(bottom-72)
    im=background.copy()
    im.paste(sprite,(691,top),sprite)
    # Render the changing arrow and velocity at high resolution for smooth text.
    overlay=Image.new('RGBA',(1024*3,1024*3),(0,0,0,0))
    d=ImageDraw.Draw(overlay)
    green=(60,180,60,255)
    tip_base=top-5
    length=max(12,min(210,22*abs(v)))
    yupper=tip_base-length
    if abs(v)>.015:
        x=727*3
        if v>0:
            start,end=tip_base*3,yupper*3
            d.line((x,start,x,end+7*3),fill=green,width=4*3)
            d.polygon([(x,end),(x-6*3,end+12*3),(x+6*3,end+12*3)],fill=green)
        else:
            start,end=yupper*3,tip_base*3
            d.line((x,start,x,end-7*3),fill=green,width=4*3)
            d.polygon([(x,end),(x-6*3,end-12*3),(x+6*3,end-12*3)],fill=green)
    label=f'v={abs(v):.1f} m/s'
    label_y=max(160,yupper-13)
    # Right align labels to leave the arrow unobstructed as magnitudes change.
    width=d.textlength(label,font=font)
    d.text((703*3-width,label_y*3),label,font=font,fill=green,anchor='lt')
    overlay=overlay.resize((1024,1024),Image.Resampling.LANCZOS)
    im.paste(overlay,(0,0),overlay)
    return im

cmd=['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','18','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT)]
p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
for i in range(192):
    p.stdin.write(np.asarray(frame(i),dtype=np.uint8).tobytes())
p.stdin.close()
if p.wait()!=0:
    raise RuntimeError('ffmpeg failed')
print(OUT)
print('Final height and velocity:',state(191/16))
