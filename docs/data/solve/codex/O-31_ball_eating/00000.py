from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import cv2
import subprocess

ROOT=Path('/app')
OUT=ROOT/'output'
OUT.mkdir(exist_ok=True)
original=np.array(Image.open(ROOT/'first_frame.png').convert('RGB'))
colors={'pink':(255,105,180),'green':(60,179,113),'orange':(255,140,0),'red':(220,20,60)}
masks={k:np.all(original==v,axis=2) for k,v in colors.items()}
black=np.all(original==0,axis=2)
# Each meal makes the eater large enough for the next ball.
# Curved detours keep it clear of balls that it cannot yet eat.
stages=[
 (0,24,[(150,191),(340,120),(562,79)],24.5,34,'pink'),
 (24,54,[(562,79),(715,260),(720,580),(540,760),(403,850)],34,56,'green'),
 (54,82,[(403,850),(690,735),(770,480),(760,250),(650,93)],56,85,'orange'),
 (82,103,[(650,93),(602,205),(469,360)],85,112,'red')]

def smooth(t):
 return t*t*(3-2*t)

def on_path(points,t):
 p=np.array(points,dtype=float)
 lengths=np.linalg.norm(np.diff(p,axis=0),axis=1)
 d=t*lengths.sum()
 for i,length in enumerate(lengths):
  if d<=length:return p[i]+(p[i+1]-p[i])*d/length
  d-=length
 return p[-1]

def frame(n):
 if n==0:return original.copy()
 canvas=original.copy();canvas[black]=255
 eaten=[]
 for a,b,points,r0,r1,meal in stages:
  if n>=b:
   eaten.append(meal)
   center=np.array(points[-1]);radius=r1
  else:
   t=max(0,(n-a)/(b-a))
   center=on_path(points,smooth(t))
   growth=smooth(np.clip((t-.72)/.28,0,1))
   radius=r0+(r1-r0)*growth
   break
 for meal in eaten:canvas[masks[meal]]=255
 # Antialias only the moving ball's boundary; untouched pixels stay exact.
 x,y=center
 x0=max(0,int(x-radius-2));y0=max(0,int(y-radius-2))
 x1=min(1024,int(x+radius+3));y1=min(1024,int(y+radius+3))
 scale=4
 alpha=Image.new('L',((x1-x0)*scale,(y1-y0)*scale),0)
 ImageDraw.Draw(alpha).ellipse(((x-radius-x0)*scale,(y-radius-y0)*scale,(x+radius-x0)*scale,(y+radius-y0)*scale),fill=255)
 alpha=np.array(alpha.resize((x1-x0,y1-y0),Image.Resampling.LANCZOS))/255.
 canvas[y0:y1,x0:x1]=np.rint(canvas[y0:y1,x0:x1]*(1-alpha[:,:,None])).astype(np.uint8)
 return canvas

command=['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','17','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')]
p=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
for n in range(108):p.stdin.write(frame(n).tobytes())
p.stdin.close()
if p.wait():raise RuntimeError('ffmpeg failed')
