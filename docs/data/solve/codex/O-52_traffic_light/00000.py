from PIL import Image, ImageDraw, ImageFont
import numpy as np
import subprocess
from pathlib import Path

ROOT=Path('/app')
base=np.array(Image.open(ROOT/'first_frame.png').convert('RGB'))
# The initial yellow is the phase following red in the specified cycle.
lights=[(512,220,451,277,1),(512,804,451,861,2),(804,512,743,569,4),(220,512,159,569,1)]
templates={1:base[277:400,451:574].copy(),2:base[861:984,451:574].copy(),4:base[569:692,743:866].copy()}
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',100)
patch=Image.new('RGB',(123,123),'white')
d=ImageDraw.Draw(patch)
b=font.getbbox('3')
d.text(((123-(b[2]-b[0]))/2-b[0],23-b[1]),'3',font=font,fill='black')
templates[3]=np.array(patch)
yy,xx=np.indices(base.shape[:2])
masks=[]
for cx,cy,*_ in lights:
 masks.append((abs(xx-cx)<70)&(abs(yy-cy)<70)&np.all(base==[255,204,0],axis=2))

def state(initial,t):
 if t<initial: return (255,204,0),initial-t
 if t<initial+4: return (0,204,0),initial+4-t
 if t<initial+8: return (255,204,0),initial+8-t
 return (255,0,0),initial+12-t

def frame(i):
 if i==0: return base
 t=min(i//16,6)
 arr=base.copy()
 for light,mask in zip(lights,masks):
  cx,cy,x,y,initial=light
  color,count=state(initial,t)
  arr[mask]=color
  if count!=initial:
   arr[y:y+123,x:x+123]=templates[count]
 return arr

(ROOT/'output').mkdir(exist_ok=True)
p=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','0','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(ROOT/'output/video.mp4')],stdin=subprocess.PIPE)
for i in range(112):
 p.stdin.write(frame(i).tobytes())
p.stdin.close()
if p.wait(): raise RuntimeError('ffmpeg failed')
