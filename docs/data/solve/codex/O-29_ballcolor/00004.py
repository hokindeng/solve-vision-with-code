from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2, subprocess, os

ROOT='/app'
im=Image.open(ROOT+'/first_frame.png').convert('RGB')
a=np.array(im)
m=((a.max(2).astype(int)-a.min(2).astype(int))>70).astype('uint8')
n,l,stats,centers=cv2.connectedComponentsWithStats(m)
groups={k:[] for k in 'ABCD'}
sprites={}
base=im.copy(); dr=ImageDraw.Draw(base)
for st,ct in zip(stats[1:],centers[1:]):
    x,y,w,h,area=st
    if area<500: continue
    cx,cy=map(int,ct); r,g,b=a[cy,cx]
    key='A' if r>200 else ('B' if g>100 else ('C' if r>30 else 'D'))
    groups[key].append((cx,cy))
    sprites[key]=im.crop((cx-23,cy-23,cx+24,cy+24))
    dr.rectangle((cx-23,cy-23,cx+23,cy+23),fill=(240,240,240))
rects={'A':(560,505,674,548),'B':(437,229,550,272),'C':(288,538,400,582),'D':(637,830,787,873)}
for rect in rects.values(): dr.rectangle(rect,fill=(240,240,240))
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',50)
def grid(n,center):
    cols=3 if n<=7 else (4 if n<=13 else 5)
    rows=(n+cols-1)//cols
    return np.array([(center[0]+(i%cols-(cols-1)/2)*53, center[1]+(i//cols-(rows-1)/2)*53) for i in range(n)],float)
def smooth(t):
    t=np.clip(t,0,1); return t*t*(3-2*t)
def label(out,pts,count):
    x=int(np.max(pts[:,0])+56); y=int(np.mean(pts[:,1]))
    d=ImageDraw.Draw(out); txt=f'A: {count}'; box=d.textbbox((x,y),txt,font=font,anchor='lm')
    d.rectangle((box[0]-3,box[1]+1,box[2]+3,box[3]+1),fill='white')
    d.text((x,y),txt,font=font,fill='black',anchor='lm')
# Each approach is followed by absorption and a short settled pause.
stages=[]; prev=np.array(groups['A'],float)
for key,center,near in [('B',(440,270),(488,245)),('C',(280,520),(335,510)),('D',(510,745),(520,650))]:
    other=np.array(groups[key],float)
    approach=prev+(np.array(near)-prev.mean(0))
    target=grid(len(prev)+len(other),center)
    stages.append((key,prev,approach,other,target))
    prev=target
os.makedirs(ROOT+'/output',exist_ok=True)
p=subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',ROOT+'/output/video.mp4'],stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
for f in range(75):
    if f<4:
        out=im.copy()
    else:
        idx=min((f-4)//23,2); t=min((f-4-idx*23)/22,1)
        key,old,near,other,target=stages[idx]
        absorbed=t>=0.45
        out=base.copy()
        # Unabsorbed clusters and labels retain their source pixels.
        for j,k in enumerate('BCD'):
            if j>idx or (j==idx and not absorbed):
                for x,y in groups[k]: out.paste(sprites[k],(x-23,y-23))
                box=rects[k]; out.paste(im.crop(box),box[:2])
        if not absorbed:
            pts=old+(near-old)*smooth(t/0.45)
            count=len(old)
        else:
            start=np.concatenate([near,other]); pts=start+(target-start)*smooth((t-0.45)/0.45)
            count=len(target)
        for x,y in pts: out.paste(sprites['A'],(round(x)-23,round(y)-23))
        label(out,pts,count)
    p.stdin.write(np.array(out).tobytes())
p.stdin.close()
if p.wait()!=0: raise RuntimeError('ffmpeg failed')
