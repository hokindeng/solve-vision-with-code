from PIL import Image, ImageDraw, ImageFont
import numpy as np
import subprocess
from pathlib import Path

ROOT=Path('/app'); OUT=ROOT/'output'; OUT.mkdir(exist_ok=True)
im=Image.open(ROOT/'first_frame.png').convert('RGB')
bg=(240,240,240)
centers={'A':[(94,257),(159,257),(94,322)],'B':[(513,832),(513,897)],'C':[(360,510),(425,510),(360,575),(425,575)],'D':[(122,778),(187,778),(122,843),(187,843),(122,908)]}
labels={'A':(199,285,313,329),'B':(572,859,685,902),'C':(466,537,578,581),'D':(254,839,370,882)}
base=im.copy(); d=ImageDraw.Draw(base)
for key,pts in centers.items():
    for x,y in pts:d.rectangle((x-28,y-28,x+28,y+28),fill=bg)
    x0,y0,x1,y1=labels[key]; d.rectangle((x0,y0,x1-1,y1-1),fill=bg)
sprites={}
for key,pts in centers.items():
    x,y=pts[0];sprites[key]=im.crop((x-28,y-28,x+29,y+29))
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',48)
def grid(n,anchor,cols):
    return [(anchor[0]+65*(i%cols),anchor[1]+65*(i//cols)) for i in range(n)]
def interp(a,b,t):
    t=t*t*(3-2*t)
    return [(x+(u-x)*t,y+(v-y)*t) for (x,y),(u,v) in zip(a,b)]
def shift(pts,anchor):
    x,y=pts[0];return [(u+anchor[0]-x,v+anchor[1]-y) for u,v in pts]
a0=centers['A'];a1=shift(a0,(448,702));a2=grid(5,(448,767),2)
a3=shift(a2,(295,315));a4=grid(9,(295,445),3)
a5=shift(a4,(56,583));a6=grid(14,(94,648),4)
def state(f):
    if f<=5:return a0,3,['B','C','D']
    if f<=22:return interp(a0,a1,(f-5)/17),3,['B','C','D']
    if f<=28:return interp(a1+centers['B'],a2,(f-22)/6),5,['C','D']
    if f<=46:return interp(a2,a3,(f-28)/18),5,['C','D']
    if f<=53:return interp(a3+centers['C'],a4,(f-46)/7),9,['D']
    if f<=71:return interp(a4,a5,(f-53)/18),9,['D']
    if f<=79:return interp(a5+centers['D'],a6,(f-71)/8),14,[]
    return a6,14,[]
def render(f):
    if f==0:return im.copy()
    frame=base.copy();pts,n,others=state(f)
    for key in others:
        for x,y in centers[key]:frame.paste(sprites[key],(x-28,y-28))
        box=labels[key];frame.paste(im.crop(box),box[:2])
    for x,y in pts:frame.paste(sprites['A'],(round(x)-28,round(y)-28))
    if n==3:
        x,y=pts[0];frame.paste(im.crop(labels['A']),(round(x+105),round(y+28)))
    else:
        x=round(max(p[0] for p in pts)+41);y=round((min(p[1] for p in pts)+max(p[1] for p in pts))/2-20)
        draw=ImageDraw.Draw(frame);s=f'A: {n}';box=draw.textbbox((x,y),s,font=font,anchor='lt')
        draw.rectangle((box[0]-3,box[1]-2,box[2]+3,box[3]+2),fill='white');draw.text((x,y),s,font=font,fill='black',anchor='lt')
    return frame
cmd=['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','0','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')]
p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
for f in range(84):p.stdin.write(np.asarray(render(f)).tobytes())
p.stdin.close();err=p.stderr.read();rc=p.wait()
if rc:raise RuntimeError(err.decode())
render(83).save(OUT/'last_frame.png')
