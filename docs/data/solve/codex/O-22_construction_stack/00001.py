from PIL import Image, ImageDraw, ImageFont
import numpy as np
import subprocess, os
from collections import deque

ROOT='/app'
im=Image.open(ROOT+'/first_frame.png').convert('RGB')
bg=im.getpixel((0,0))
# Preserve the supplied artwork, including its antialiased text and block details.
base=im.copy()
d=ImageDraw.Draw(base)
for box in [(79,695,177,825),(207,759,305,825),(335,695,433,825)]: d.rectangle(box,fill=bg)
colors={'B':(52,152,219),'O':(230,126,34),'R':(231,76,60),'Y':(241,196,15),'P':(155,89,182)}
sources={'B':(591,759),'Y':(719,759),'R':(207,759),'O':(79,759),'P':(335,695)}
sprites={}
for name,(x,y) in sources.items():
 a=np.array(im.crop((x,y,x+99,y+67)))
 # A stacked block shares its outline with the block underneath. Restore
 # its standalone lower outline from an isolated block before it moves.
 if name=='P':
  edge=np.array(im.crop((591,823,690,826)))
  for old,new in [(colors['B'],colors[name]),(tuple(max(0,c-40) for c in colors['B']),tuple(max(0,c-40) for c in colors[name]))]:
   edge[np.all(edge==old,axis=2)]=new
  a[-3:]=edge
 sprites[name]=Image.fromarray(a)
start=(('O','B'),('R',),('Y','P'))
goal=(('B',),('Y',),('R','P','O'))
q=deque([(start,[])]);seen={start}
while q:
 state,moves=q.popleft()
 if state==goal: break
 for i in range(3):
  if not state[i]:continue
  for j in range(3):
   if i==j:continue
   n=list(map(list,state));b=n[i].pop();n[j].append(b);n=tuple(map(tuple,n))
   if n not in seen:seen.add(n);q.append((n,moves+[(i,j,b)]))
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',32)
def draw(st,count,moving=None):
 frame=base.copy()
 for col,stack in enumerate(st):
  for level,b in enumerate(stack):frame.paste(sprites[b],(79+128*col,759-64*level))
 if moving:
  b,x,y=moving;frame.paste(sprites[b],(round(x),round(y)))
 if count:
  d=ImageDraw.Draw(frame);d.rectangle((425,977,598,1014),fill=bg)
  txt=f'Moves: {count}';d.text((512,975),txt,font=font,fill=(50,50,50),anchor='mt')
 return frame
os.makedirs(ROOT+'/output',exist_ok=True)
p=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',ROOT+'/output/video.mp4'],stdin=subprocess.PIPE)
def emit(f):p.stdin.write(np.asarray(f).tobytes())
for _ in range(7):emit(im)
st=list(map(list,start))
for count,(i,j,b) in enumerate(moves,1):
 sy=759-64*(len(st[i])-1);dy=759-64*len(st[j]);sx=79+128*i;dx=79+128*j
 st[i].pop();high=min(sy,dy,759-64*max(map(len,st)))-85
 for k in range(17):
  t=(k+1)/17
  ease=lambda v:v*v*(3-2*v)
  if t<0.28:x=sx;y=sy+(high-sy)*ease(t/.28)
  elif t<.72:x=sx+(dx-sx)*ease((t-.28)/.44);y=high
  else:x=dx;y=high+(dy-high)*ease((t-.72)/.28)
  if k==16:
   st[j].append(b);emit(draw(st,count))
  else:emit(draw(st,count-1,(b,x,y)))
for _ in range(7):emit(draw(st,len(moves)))
p.stdin.close()
if p.wait():raise RuntimeError('ffmpeg failed')
assert tuple(map(tuple,st))==goal
print('Created 150 frames; legal moves:',moves)
