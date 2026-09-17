from PIL import Image, ImageDraw, ImageFont
import numpy as np
import subprocess
from pathlib import Path

ROOT=Path('/app'); (ROOT/'output').mkdir(exist_ok=True)
original=Image.open(ROOT/'first_frame.png').convert('RGB')
bg=original.getpixel((0,100))
orange=original.crop((79,759,178,826))
sprites={'O':orange}
for name, color, crop in [('G',(46,204,113),(335,695,434,762)),('P',(155,89,182),(335,759,434,826))]:
    template=np.array(orange).copy()
    for src,dst in [((230,126,34),color),((255,166,74),tuple(min(255,c+40) for c in color)),((190,86,0),tuple(max(0,c-40) for c in color))]:
        template[np.all(template==src,axis=2)]=dst
    # Preserve the original colored body and letter; restore the unobscured top edge.
    block=np.array(original.crop(crop)).copy()
    block[:5]=template[:5]
    sprites[name]=Image.fromarray(block)
base=original.copy(); draw=ImageDraw.Draw(base)
draw.rectangle((79,759,177,825),fill=bg)
draw.rectangle((335,695,433,825),fill=bg)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',32)
xs=[79,207,335]
stacks=[['O'],[],['P','G']]
moves=[(2,0),(2,1),(0,1),(0,1)]

def render(stacks, count, moving=None):
    im=base.copy()
    for col, stack in enumerate(stacks):
        for level,name in enumerate(stack):
            im.paste(sprites[name],(xs[col],759-64*level))
    if moving:
        name,x,y=moving; im.paste(sprites[name],(round(x),round(y)))
    if count:
        d=ImageDraw.Draw(im)
        d.rectangle((570,978,595,1010),fill=bg)
        d.text((570,975),str(count),font=font,fill=(50,50,50))
    return im

def ease(t):return t*t*(3-2*t)
frames=[original.copy() for _ in range(5)]
for count,(src,dst) in enumerate(moves):
    name=stacks[src].pop()
    x0,y0=xs[src],759-64*len(stacks[src])
    x1,y1=xs[dst],759-64*len(stacks[dst])
    high=min(y0,y1)-100
    for k in range(18):
        t=(k+1)/18
        if t<.28:
            x=x0;y=y0+(high-y0)*ease(t/.28)
        elif t<.72:
            x=x0+(x1-x0)*ease((t-.28)/.44);y=high
        else:
            x=x1;y=high+(y1-high)*ease((t-.72)/.28)
        frames.append(render(stacks,count,(name,x,y)))
    stacks[dst].append(name)
    for _ in range(2):frames.append(render(stacks,count+1))
final=render(stacks,4)
# The finished arrangement reproduces the target artwork exactly.
final.paste(original.crop((719,631,818,826)),(207,631))
while len(frames)<90:frames.append(final.copy())
proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(ROOT/'output/video.mp4')],stdin=subprocess.PIPE)
for im in frames:proc.stdin.write(im.tobytes())
proc.stdin.close()
if proc.wait()!=0:raise RuntimeError('ffmpeg failed')
final.save(ROOT/'output/last_frame.png')
