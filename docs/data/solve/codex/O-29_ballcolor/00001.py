from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os

ROOT='/app'
source=Image.open(ROOT+'/first_frame.png').convert('RGB')
bg=(240,240,240)
base=source.copy()
a=np.array(source)
mask=np.any(a!=240,axis=2).astype(np.uint8)
n,_,stats,_=cv2.connectedComponentsWithStats(mask)
d=ImageDraw.Draw(base)
for x,y,w,h,area in stats[1:]:
    if y>100: d.rectangle((int(x),int(y),int(x+w-1),int(y+h-1)),fill=bg)
ball=source.crop((447,476,491,520))
alpha=Image.fromarray((np.any(np.array(ball)!=240,axis=2)*255).astype('uint8'))
clusters={
 'B': [(193,338),(193,391)],
 'C': [(483,784),(535,784),(483,836),(535,836)],
 'D': [(219,714),(272,714),(219,766),(272,766),(219,819)]}
regions={'B':(171,316,354,413),'C':(461,762,680,858),'D':(197,692,442,841)}
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',50)

def layout(n,x,y,cols):
    return np.array([(x+53*(i%cols),y+53*(i//cols)) for i in range(n)],dtype=float)

def ease(t):
    t=np.clip(t,0,1)
    return t*t*(3-2*t)

initial=np.array([(469,498),(522,498),(469,551)],dtype=float)
stages=[
 (5,18,26,'B',layout(3,270,338,2),layout(5,193,338,3)),
 (29,42,50,'C',layout(5,483,625,3),layout(9,430,678,3)),
 (53,66,74,'D',layout(9,300,610,3),layout(14,219,610,4))]

def frame(f):
    if f<5:return source.copy()
    pos=initial.copy();remaining=set(clusters);count=3
    for start,contact,end,name,arrival,merged in stages:
        if f<start:break
        if f<contact:
            pos=pos+(arrival-pos)*ease((f-start)/(contact-start))
            break
        remaining.remove(name)
        combined=np.concatenate([arrival,np.array(clusters[name],float)])
        count=len(combined)
        pos=combined+(merged-combined)*ease((f-contact)/(end-contact))
        if f<end:break
        pos=merged.copy()
    im=base.copy()
    for name in remaining:
        box=regions[name];im.paste(source.crop(box),box[:2])
    for x,y in pos:
        im.paste(ball,(round(x)-22,round(y)-22),alpha)
    if count==3:
        label=source.crop((555,519,669,563))
    else:
        text=f'A: {count}'
        bounds=font.getbbox(text)
        label=Image.new('RGB',(bounds[2]+6,bounds[3]-bounds[1]+7),'white')
        ImageDraw.Draw(label).text((3,3-bounds[1]),text,font=font,fill='black')
    x=round(pos[:,0].max()+33)
    y=round((pos[:,1].min()+pos[:,1].max())/2-5)
    im.paste(label,(x,y))
    return im

os.makedirs(ROOT+'/output',exist_ok=True)
import subprocess
proc=subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',ROOT+'/output/video.mp4'],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
for f in range(80):proc.stdin.write(np.asarray(frame(f)).tobytes())
proc.stdin.close()
err=proc.stderr.read();proc.wait()
if proc.returncode:raise RuntimeError(err.decode())
assert np.array_equal(np.asarray(frame(0)),np.asarray(source))
frame(79).save(ROOT+'/output/last_frame.png')
