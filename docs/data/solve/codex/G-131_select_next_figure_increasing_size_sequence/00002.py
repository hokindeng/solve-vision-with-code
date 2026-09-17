from PIL import Image, ImageDraw, ImageFont
import numpy as np
import subprocess, os, math

ROOT='/app'
os.makedirs(ROOT+'/output',exist_ok=True)
base=Image.open(ROOT+'/first_frame.png').convert('RGB')
S=3
font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def font(n): return ImageFont.truetype(font_path,n*S)
red=(220,35,40,255)
ink=(85,85,85,255)
def render(i):
    if i==0: return np.asarray(base)
    overlay=Image.new('RGBA',(1024*S,1024*S))
    d=ImageDraw.Draw(overlay)
    def line(points,fill=red,width=2): d.line([(int(x*S),int(y*S)) for x,y in points],fill=fill,width=width*S)
    def text(x,y,s,n=20,fill=ink): d.text((x*S,y*S),s,font=font(n),fill=fill,anchor='mm')
    def arrow(x1,x2,y,t):
        x=x1+(x2-x1)*min(1,t)
        line([(x1,y),(x,y)])
        if t>=1: line([(x-7,y-5),(x,y),(x-7,y+5)])
    if i>=5:
        text(512,205,'Each side grows by about 16 px',23)
        arrow(255,400,263,(i-5)/9)
        if i>=14: text(328,241,'+ size step',17)
    if i>=18:
        arrow(443,573,263,(i-18)/9)
        if i>=27: text(508,241,'+ size step',17)
    if i>=30:
        arrow(633,744,263,(i-30)/8)
        if i>=38:
            text(692,241,'+ size step',17)
            text(782,418,'76 px',19)
            text(512,510,'Next: a 76 px black square',23)
    if i>=40:
        # Complete the sequence inside the existing dashed target.
        d.rectangle((744*S,300*S,820*S-1,376*S-1),fill=(0,0,0,255))
    if i>=43:
        p=min(1,(i-43)/12)
        d.arc((565*S,804*S,689*S,926*S),start=-90,end=-90+360*p,fill=red,width=4*S)
    if i>=55:
        text(626,983,'Correct',19,red)
    overlay=overlay.resize(base.size,Image.Resampling.LANCZOS)
    return np.asarray(Image.alpha_composite(base.convert('RGBA'),overlay).convert('RGB'))

cmd=['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-preset','slow','-crf','15','-pix_fmt','yuv420p','-movflags','+faststart',ROOT+'/output/video.mp4']
p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
for i in range(60): p.stdin.write(render(i).tobytes())
p.stdin.close()
err=p.stderr.read(); rc=p.wait()
if rc: raise RuntimeError(err.decode())
assert np.array_equal(render(0),np.asarray(base))
print('Created output/video.mp4: 60 frames, 1024x1024, 16 fps')
