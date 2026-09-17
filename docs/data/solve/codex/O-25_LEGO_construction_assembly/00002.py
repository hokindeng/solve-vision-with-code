from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2
import subprocess

ROOT=Path('/app')
OUT=ROOT/'output'
OUT.mkdir(exist_ok=True)
source=Image.open(ROOT/'first_frame.png').convert('RGB')
a=np.array(source)
# Silhouette of the brick in the instruction callout.
outline=[(30,402),(204,302),(290,352),(290,412),(117,511),(30,462)]
mask=Image.new('L',source.size,0)
ImageDraw.Draw(mask).polygon(outline,fill=255)
m=np.array(mask)>0
# Include the one-pixel outlines and the exact original raster silhouette.
ai=a.astype(np.int16)
blue=(ai[:,:,2]>ai[:,:,0]+35)&(ai[:,:,2]>ai[:,:,1]+25)
blue[520:]=False
blue[:,300:]=False
m|=blue
mask=Image.fromarray((m*255).astype('uint8')).filter(ImageFilter.MaxFilter(3))
m=np.array(mask)>0

# Restore only the area vacated by the brick, including the hidden callout.
restored=Image.new('RGB',source.size,(245,245,240))
r=ImageDraw.Draw(restored)
r.rectangle((70,331,240,502),fill=(180,180,180))
r.rectangle((72,333,238,500),fill='white')
base=source.copy()
base.paste(restored,(0,0),mask)
# The instruction arrow is a fixed annotation, including its white keyline.
yy,xx=np.indices(m.shape)
annotation=m & (xx>=155)&(yy>=413)&(yy<=434)&(((a[:,:,0]>240)&(a[:,:,1]<40))|np.all(a==255,axis=2))
b=np.array(base)
b[annotation]=a[annotation]
base=Image.fromarray(b)

# Recover the small surface area originally obscured by the arrow.
clean=Image.new('RGB',source.size)
d=ImageDraw.Draw(clean)
d.polygon([(30,402),(117,451),(117,511),(30,462)],fill=(0,59,133))
d.polygon([(30,462),(74,437),(117,451),(117,511)],fill=(0,59,133))
d.polygon([(117,451),(290,352),(290,412),(117,511)],fill=(0,72,162))
d.polygon([(30,402),(204,302),(290,352),(117,451)],fill=(0,102,229))
sprite=a.copy()
sprite[annotation]=np.array(clean)[annotation]
rgba=np.dstack((sprite,(m*255).astype('uint8')))
brick=Image.fromarray(rgba).crop((29,301,292,513))

frames=[]
for frame in range(46):
    if frame==0:
        im=source.copy()
    else:
        # A slow takeoff, a lifted transfer, then a short downward seating motion.
        t=min(1,max(0,(frame-2)/39))
        s=t*t*(3-2*t)
        dx=478*s
        dy=261*s-65*math.sin(math.pi*t)-12*s
        if frame>=41:
            u=min(1,(frame-41)/3)
            dy=249+12*(u*u*(3-2*u))
            dx=478
        im=base.copy()
        im.paste(brick,(29+round(dx),301+round(dy)),brick)
    frames.append(np.array(im))
# Feed RGB frames directly to ffmpeg to avoid intermediate image files.
proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size','1024x1024','-framerate','16','-i','-','-an','-c:v','libx264','-preset','slow','-crf','0','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')],stdin=subprocess.PIPE)
for f in frames:
    proc.stdin.write(f.tobytes())
proc.stdin.close()
if proc.wait()!=0:
    raise RuntimeError('Video encoding failed')
Image.fromarray(frames[-1]).save(OUT/'last_frame.png')
