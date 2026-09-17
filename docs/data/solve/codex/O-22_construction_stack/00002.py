from PIL import Image, ImageDraw, ImageFont
import numpy as np
import subprocess
from pathlib import Path

ROOT=Path('/app')
im=Image.open(ROOT/'first_frame.png').convert('RGB')
bg=(245,245,250)
positions={'O':(79,695),'R':(79,759),'B':(207,695),'Y':(207,759),'G':(335,759)}
colors={'O':(230,126,34),'R':(231,76,60),'B':(52,152,219),'Y':(241,196,15),'G':(46,204,113)}
template=np.array(im.crop((335,759,434,826)))
sprites={}
for name,(x,y) in positions.items():
 a=template.copy()
 color=colors[name]
 for old,new in [((46,204,113),color),((86,244,153),tuple(min(255,c+40) for c in color)),((6,164,73),tuple(max(0,c-40) for c in color))]:
  a[np.all(template==old,axis=2)]=new
 # Replace the complete lettering region with the original pixels.
 a[8:59,10:89]=np.array(im.crop((x+10,y+8,x+89,y+59)))
 sprites[name]=Image.fromarray(a)
base=im.copy()
d=ImageDraw.Draw(base)
for x in (79,207,335):d.rectangle((x,695,x+98,825),fill=bg)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',32)
moves=[(1,2),(1,2),(0,1),(2,1),(2,1),(0,1)]
state=[['R','O'],['Y','B'],['G']]
xs=[79,207,335]
def draw(st,moving=None,count=0):
 frame=base.copy()
 for i,stack in enumerate(st):
  for level,b in enumerate(stack):frame.paste(sprites[b],(xs[i],759-64*level))
 if moving:
  b,x,y=moving
  frame.paste(sprites[b],(round(x),round(y)))
 if count:
  pen=ImageDraw.Draw(frame)
  pen.rectangle((425,976,600,1013),fill=bg)
  pen.text((512,981),f'Moves: {count}',font=font,fill=(50,50,50),anchor='mt')
 return frame
out=ROOT/'output';out.mkdir(exist_ok=True)
proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')],stdin=subprocess.PIPE)
frames=[]
frames.extend([im.copy() for _ in range(6)])
for n,(src,dst) in enumerate(moves):
 b=state[src].pop(); x0=xs[src];y0=759-64*len(state[src]);x1=xs[dst];y1=759-64*len(state[dst]);high=min(y0,y1)-100
 for k in range(17):
  t=k/16
  if t<.3:
   u=t/.3;u=u*u*(3-2*u);x=x0;y=y0+(high-y0)*u
  elif t<.7:
   u=(t-.3)/.4;u=u*u*(3-2*u);x=x0+(x1-x0)*u;y=high
  else:
   u=(t-.7)/.3;u=u*u*(3-2*u);x=x1;y=high+(y1-high)*u
  frames.append(draw(state,(b,x,y),n+(k==16)))
 state[dst].append(b)
 frames.append(draw(state,count=n+1))
frames.extend([draw(state,count=6) for _ in range(6)])
assert len(frames)==120
assert state==[[],['O','Y','B','R'],['G']]
for f in frames:proc.stdin.write(f.tobytes())
proc.stdin.close()
assert proc.wait()==0
