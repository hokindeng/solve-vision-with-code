from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
# All drawing is confined to the missing answer's bounding box.
x0, y0, x1, y1 = 776, 680, 953, 857
original = base[y0:y1, x0:x1].copy()
w, h = x1-x0, y1-y0
scale = 4
points = [(784,729),(864,729),(864,689),(944,769),(864,849),(864,809),(784,809),(784,729)]
points = [(x-x0,y-y0) for x,y in points]

def outline(dashed):
    im = Image.new('RGB',(w*scale,h*scale),'white')
    d = ImageDraw.Draw(im)
    color = (8, 102, 218) if dashed else (12, 98, 193)
    if not dashed:
        d.line([(x*scale,y*scale) for x,y in points],fill=color,width=8*scale,joint='curve')
    else:
        # Carry dash phase around the perimeter for a consistent patterned outline.
        distance=0.0
        for a,b in zip(points, points[1:]):
            a,b=np.array(a,dtype=float),np.array(b,dtype=float)
            length=float(np.linalg.norm(b-a))
            pos=0.0
            while pos < length-1e-6:
                phase=(distance+pos)%8
                step=min(length-pos, (5-phase) if phase<5 else (8-phase))
                if step<1e-6: step=min(length-pos,1e-5)
                if phase<5:
                    p=a+(b-a)*pos/length
                    q=a+(b-a)*(pos+step)/length
                    d.line([tuple(p*scale),tuple(q*scale)],fill=color,width=3*scale)
                pos+=step
            distance+=length
    return np.array(im.resize((w,h),Image.Resampling.LANCZOS)).astype(float)

solid=outline(False)
dashed=outline(True)
def smooth(t):
    t=np.clip(t,0,1)
    return t*t*(3-2*t)

proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-preset','slow','-crf','0','-pix_fmt','yuv420p',str(OUT/'video.mp4')],stdin=subprocess.PIPE)
for i in range(60):
    frame=base.copy()
    if i:
        erase=smooth(i/12)
        patch=original.astype(float)*(1-erase)+255*erase
        appear=smooth((i-10)/15)
        transform=smooth((i-26)/29)
        shape=solid*(1-transform)+dashed*transform
        patch=patch*(1-appear)+shape*appear
        frame[y0:y1,x0:x1]=np.clip(np.rint(patch),0,255).astype(np.uint8)
    proc.stdin.write(frame.tobytes())
proc.stdin.close()
if proc.wait()!=0:
    raise RuntimeError('ffmpeg failed')
