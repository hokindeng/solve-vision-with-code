from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path('/app')
BASE=Image.open(ROOT/'first_frame.png').convert('RGB')
A=np.asarray(BASE)
# The south yellow is the Red -> Yellow -> Green phase of the stated cycle.
COLORS=[(255,0,0),(255,200,0),(0,200,0),(255,200,0)]
LIGHTS=[((512,220),(450,276,575,400),0,3),
        ((512,804),(450,860,575,984),1,4),
        ((804,512),(743,568,867,692),2,1),
        ((220,512),(159,568,283,692),2,1)]
MASKS=[]
for (cx,cy),box,phase,remaining in LIGHTS:
    yy,xx=np.indices(A.shape[:2])
    MASKS.append((np.max(np.abs(A.astype(int)-np.array(COLORS[phase])),axis=2)==0)&(abs(xx-cx)<70)&(abs(yy-cy)<70))
# Reuse the source's numeral artwork wherever available.
GLYPHS={}
for digit,box in [(3,(451,277,574,399)),(1,(159,568,282,692)),(4,(451,860,574,983))]:
    crop=BASE.crop(box)
    ink=np.any(np.asarray(crop)<128,axis=2)
    ys,xs=np.where(ink)
    GLYPHS[digit]=crop.crop((int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)))
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',104)
b=font.getbbox('2')
g=Image.new('RGB',(b[2]-b[0],b[3]-b[1]),'white')
ImageDraw.Draw(g).text((-b[0],-b[1]),'2',font=font,fill='black')
ink=np.any(np.asarray(g)<128,axis=2); ys,xs=np.where(ink)
GLYPHS[2]=g.crop((int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)))

def frame(step):
    if step==0:return BASE.copy()
    arr=A.copy()
    states=[]
    for j,(_,box,p,r) in enumerate(LIGHTS):
        if step<r: phase,remain=p,r-step
        else: phase,remain=(p+1)%4,4-(step-r)
        arr[MASKS[j]]=COLORS[phase]
        states.append((box,remain))
    out=Image.fromarray(arr)
    draw=ImageDraw.Draw(out)
    for box,digit in states:
        x0,y0,x1,y1=box
        draw.rectangle((x0+1,y0+1,x1-2,y1-2),fill='white')
        glyph=GLYPHS[digit]
        # Keep the source text's visual centering inside each white placard.
        center_y=336 if y0==276 else (628 if y0==568 else 920)
        out.paste(glyph,(round((x0+x1)/2-glyph.width/2),round(center_y-glyph.height/2)))
    return out

def main():
    (ROOT/'output').mkdir(exist_ok=True)
    cmd=['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','14','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(ROOT/'output/video.mp4')]
    p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
    for i in range(80):
        p.stdin.write(frame(min(i//16,4)).tobytes())
    p.stdin.close()
    if p.wait()!=0:raise RuntimeError('ffmpeg failed')
if __name__=='__main__':main()
