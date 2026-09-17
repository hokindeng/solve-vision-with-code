from PIL import Image, ImageDraw, ImageFont
import numpy as np
import subprocess, os

ROOT='/app'
im=Image.open(ROOT+'/first_frame.png').convert('RGB')
bg=im.getpixel((0,0))
# Preserve the original artwork, using its block faces as sprites.
sources={'P':(79,631),'B':(79,695),'G':(79,759),'O':(207,695),'R':(207,759)}
sprites={}
for name,(x,y) in sources.items():
    s=im.crop((x,y,x+99,y+67)).convert('RGBA')
    # The tiny rounded corners must be isolated from adjacent blocks.
    d=ImageDraw.Draw(s)
    d.line((0,0,98,0),fill=(0,0,0,0))
    d.line((0,66,98,66),fill=(0,0,0,0))
    for xx in (1,97):
        d.point((xx,0),fill=(50,50,50,255));d.point((xx,66),fill=(50,50,50,255))
    sprites[name]=s
base=im.copy();d=ImageDraw.Draw(base)
d.rectangle((79,631,177,825),fill=bg)
d.rectangle((207,695,305,825),fill=bg)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',32)
stacks=[['G','B','P'],['R','O'],[]]
moves=[(1,2),(0,2),(1,2),(0,2),(0,1),(2,1)]
xs=[79,207,335]
def render(stacks, moving=None, count=0):
    frame=base.copy()
    for i,stack in enumerate(stacks):
        for lev,b in enumerate(stack):
            frame.paste(sprites[b],(xs[i],759-64*lev),sprites[b])
    if moving:
        b,x,y=moving;frame.paste(sprites[b],(round(x),round(y)),sprites[b])
    if count:
        d=ImageDraw.Draw(frame);d.rectangle((425,977,600,1015),fill=bg)
        d.text((512,994),f'Moves: {count}',font=font,fill=(50,50,50),anchor='mm')
    return frame
os.makedirs(ROOT+'/output',exist_ok=True)
p=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','0','-preset','medium','-pix_fmt','yuv420p',ROOT+'/output/video.mp4'],stdin=subprocess.PIPE)
def emit(frame):p.stdin.write(np.asarray(frame).tobytes())
for _ in range(6):emit(im)
for k,(src,dst) in enumerate(moves):
    y0=759-64*(len(stacks[src])-1); y1=759-64*len(stacks[dst]);b=stacks[src].pop()
    x0,x1=xs[src],xs[dst]; high=min(y0,y1)-110
    for t in range(1,17):
        u=t/16
        def ease(v):return v*v*(3-2*v)
        if u<.3:x=x0;y=y0+(high-y0)*ease(u/.3)
        elif u<.7:x=x0+(x1-x0)*ease((u-.3)/.4);y=high
        else:x=x1;y=high+(y1-high)*ease((u-.7)/.3)
        emit(render(stacks,(b,x,y),k))
    stacks[dst].append(b)
    for _ in range(2):emit(render(stacks,count=k+1))
for _ in range(6):emit(render(stacks,count=6))
p.stdin.close()
assert p.wait()==0
assert stacks==[[],['G','B'],['O','P','R']]
