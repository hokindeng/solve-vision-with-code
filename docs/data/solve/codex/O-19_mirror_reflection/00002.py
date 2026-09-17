from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path('/app')
OUT=ROOT/'output'
OUT.mkdir(exist_ok=True)
base=np.array(Image.open(ROOT/'first_frame.png').convert('RGB'))
h,w=base.shape[:2]
y,x=np.mgrid[:h,:w]
# Erase only the lettering and angle arc; retain the incident ray and normal.
gray=(base.max(axis=2)-base.min(axis=2)<8)&(base.min(axis=2)<250)
annotation=gray & (((x>=678)&(x<=764)&(y>=557)&(y<=582)) | ((x>=600)&(x<=631)&(y>=560)&(y<=578)))
incident=(base[:,:,2]>180)&(base[:,:,0]<150)&(base[:,:,1]<150)
p0=np.array([629.,603.])
direction=np.array([484.,-553.]);direction/=np.linalg.norm(direction)
length=(1023-p0[0])/direction[0]
scale=4
import subprocess
proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size','1024x1024','-framerate','16','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(OUT/'video.mp4')],stdin=subprocess.PIPE)
for i in range(35):
    frame=base.copy()
    if i:
        fade=min(i/6,1)
        frame[annotation]=np.rint(base[annotation]*(1-fade)+255*fade).astype(np.uint8)
        progress=max(0,(i-2)/32)
        if progress>0:
            tip=p0+direction*(length*progress)
            layer=Image.new('L',(w*scale,h*scale),0)
            draw=ImageDraw.Draw(layer)
            def line(a,b,width=2):
                draw.line([tuple(a*scale),tuple(b*scale)],fill=255,width=round(width*scale))
            line(p0,tip)
            if length*progress>24:
                normal=np.array([-direction[1],direction[0]])
                line(tip,tip-direction*22+normal*8)
                line(tip,tip-direction*22-normal*8)
            alpha=np.array(layer.resize((w,h),Image.Resampling.LANCZOS))/255.*.60
            alpha[(y>=600)|incident]=0
            frame=np.rint(frame*(1-alpha[:,:,None])+np.array([0,0,255])*alpha[:,:,None]).astype(np.uint8)
    proc.stdin.write(frame.tobytes())
    if i==34: Image.fromarray(frame).save(OUT/'last_frame.png')
proc.stdin.close()
if proc.wait()!=0: raise RuntimeError('ffmpeg failed')
