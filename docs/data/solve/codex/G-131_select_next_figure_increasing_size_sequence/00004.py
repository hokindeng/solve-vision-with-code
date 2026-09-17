from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os

ROOT='/app'
base=Image.open(ROOT+'/first_frame.png').convert('RGB')
os.makedirs(ROOT+'/output',exist_ok=True)
font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font=ImageFont.truetype(font_path,21)
small=ImageFont.truetype(font_path,18)
S=3
teal=(55,215,202)
ink=(85,85,85)

def make_frame(i):
    if i==0: return np.array(base)
    # All annotation drawing is restricted to a transparent overlay.
    layer=Image.new('RGBA',(1024*S,1024*S),(0,0,0,0))
    d=ImageDraw.Draw(layer)
    def line(points,fill=ink,width=2):
        d.line([(int(x*S),int(y*S)) for x,y in points],fill=fill,width=width*S)
    def text(x,y,t,size=21,fill=ink):
        f=ImageFont.truetype(font_path,size*S)
        d.text((x*S,y*S),t,font=f,fill=fill,anchor='mm')
    def measure(x,w):
        y=421
        line([(x-w/2,y),(x+w/2,y)])
        for a in [x-w/2,x+w/2]: line([(a,y-5),(a,y+5)])
        text(x,448,str(w)+' px',18)
    if i>=5: measure(241,30)
    if i>=12:
        measure(421,56)
        text(331,483,'+26 px',18)
        line([(276,483),(288,483)])
        line([(373,483),(386,483)])
        line([(381,479),(386,483),(381,487)])
    if i>=21:
        measure(602,82)
        text(512,483,'+26 px',18)
        line([(457,483),(469,483)])
        line([(554,483),(567,483)])
        line([(562,479),(567,483),(562,487)])
    if i>=30:
        text(782,448,'108 px',18)
        text(692,483,'+26 px',18)
        line([(637,483),(649,483)])
        line([(734,483),(747,483)])
        line([(742,479),(747,483),(742,487)])
        text(512,546,'Same diamond, same color: 82 + 26 = 108 px')
    if i>=38:
        frac=min(1,(i-37)/13)
        d.arc((91*S,789*S,242*S,939*S),start=-90,end=-90+360*frac,fill=(225,35,40),width=5*S)
    if i>=52:
        # The selected continuation fits within the existing dashed target.
        x,y,r=782,337,54
        d.polygon([(x*S,(y-r)*S),((x+r)*S,y*S),(x*S,(y+r)*S),((x-r)*S,y*S)],fill=teal)
    layer=layer.resize(base.size,Image.Resampling.LANCZOS)
    return np.array(Image.alpha_composite(base.convert('RGBA'),layer).convert('RGB'))

# Encode at the requested size, frame rate, pixel format and codec.
import subprocess
p=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','15','-pix_fmt','yuv420p',ROOT+'/output/video.mp4'],stdin=subprocess.PIPE)
for i in range(60):
    p.stdin.write(make_frame(i).tobytes())
p.stdin.close()
if p.wait()!=0: raise RuntimeError('ffmpeg failed')
