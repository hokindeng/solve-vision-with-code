from PIL import Image, ImageDraw, ImageFont
import numpy as np
import subprocess, os

ROOT='/app'
source=Image.open(ROOT+'/first_frame.png').convert('RGB')
BG=(240,240,240)
# Exact original ball sprites preserve the source rasterization and outline.
centers={
 'A':[(292,233),(337,233),(292,278),(337,278),(292,323)],
 'B':[(411,728),(411,773)],
 'C':[(510,449),(555,449),(510,494),(555,494),(510,539),(555,539)],
 'D':[(93+45*c,431+45*r) for r in range(4) for c in range(3)]}
labels={'A':(385,275,499,318),'B':(453,746,566,789),'C':(603,489,715,533),'D':(239,494,389,537)}
sprites={k:source.crop((v[0][0]-20,v[0][1]-20,v[0][0]+21,v[0][1]+21)) for k,v in centers.items()}
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',50)
def erase(im,k):
 d=ImageDraw.Draw(im)
 for x,y in centers[k]: d.rectangle((x-20,y-20,x+20,y+20),fill=BG)
 x0,y0,x1,y1=labels[k]; d.rectangle((x0,y0,x1-1,y1-1),fill=BG)
def ball(im,p,k='A'):
 im.paste(sprites[k],(round(p[0])-20,round(p[1])-20))
def label(im,p,n):
 if n==5:
  im.paste(source.crop(labels['A']),(round(p[0]),round(p[1])))
 else:
  d=ImageDraw.Draw(im); text=f'A: {n}'; b=d.textbbox((0,0),text,font=font)
  x,y=map(round,p); d.rectangle((x,y,x+b[2]+5,y+b[3]-b[1]+5),fill='white')
  d.text((x+2,y-b[1]+1),text,font=font,fill='black')
def smooth(t): return t*t*(3-2*t)
def blend(a,b,t): return np.asarray(a)*(1-t)+np.asarray(b)*t
# Every merger obeys strict size comparison: 5>2, 7>6, 13>12.
plans=[
 ('B',(20,405),[(321+45*(i%3),683+45*(i//3)) for i in range(7)],(453,746)),
 ('C',(100,-278),[(443+45*(i%4),404+45*(i//4)) for i in range(13)],(623,489)),
 ('D',(-220,-10),[(93+45*(i%5),431+45*(i//5)) for i in range(25)],(320,510))]
os.makedirs(ROOT+'/output',exist_ok=True)
proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',ROOT+'/output/video.mp4'],stdin=subprocess.PIPE)
def emit(im): proc.stdin.write(np.asarray(im,dtype=np.uint8).tobytes())
for _ in range(4): emit(source)
base=source.copy(); erase(base,'A')
positions=np.array(centers['A'],float); lp=np.array((385,275),float); n=5
for target,offset,finalpos,finallabel in plans:
 assert len(centers[target])<n
 arrival=positions+offset
 arrival_label=lp+offset
 for j in range(14):
  t=smooth((j+1)/14); im=base.copy()
  for p in blend(positions,arrival,t): ball(im,p)
  label(im,blend(lp,arrival_label,t),n); emit(im)
 erase(base,target)
 starts=np.concatenate([arrival,np.array(centers[target],float)])
 finalpos=np.array(finalpos,float)
 for j in range(10):
  t=smooth((j+1)/10); im=base.copy()
  for idx,p in enumerate(blend(starts,finalpos,t)):
   color='A' if idx<n or t>=0.5 else target
   ball(im,p,color)
  label(im,blend(arrival_label,finallabel,t),n+len(centers[target]) if j>=5 else n)
  emit(im)
 positions=finalpos; lp=np.array(finallabel,float); n+=len(centers[target])
for _ in range(4): emit(im)
proc.stdin.close()
if proc.wait()!=0: raise RuntimeError('ffmpeg failed')
