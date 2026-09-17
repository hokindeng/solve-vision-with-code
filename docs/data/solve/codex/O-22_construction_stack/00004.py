from PIL import Image, ImageDraw, ImageFont
import numpy as np
import subprocess
from pathlib import Path

ROOT=Path('/app')
original=Image.open(ROOT/'first_frame.png').convert('RGB')
bg=(245,245,250)
base=original.copy()
d=ImageDraw.Draw(base)
d.rectangle((79,631,177,825),fill=bg)
d.rectangle((207,695,305,825),fill=bg)
# Preserve the exact raster artwork of every block, using an isolated block's silhouette.
mask=np.any(np.array(original.crop((847,759,946,826)))!=bg,axis=2)
locations={'G':(79,631),'Y':(79,695),'P':(79,759),'B':(207,695),'R':(207,759)}
sprites={}
for name,(x,y) in locations.items():
    tile=original.crop((x,y,x+99,y+67)).convert('RGBA')
    tile.putalpha(Image.fromarray((mask*255).astype('uint8')))
    sprites[name]=tile
moves=[(1,2,'B'),(1,2,'R'),(0,1,'G'),(0,1,'Y'),(0,1,'P'),(2,0,'R'),(1,0,'P')]
initial=[['P','Y','G'],['R','B'],[]]
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',32)

def position(stack,level):
    return 79+128*stack,759-64*level

def frame(index):
    if index==0:return original.copy()
    # Seven 18-frame moves, with opening and closing holds (135 frames total).
    elapsed=max(0,index-4)
    completed=min(7,elapsed//18)
    phase=(elapsed%18)/17
    stacks=[s[:] for s in initial]
    for src,dst,block in moves[:completed]:
        assert stacks[src].pop()==block
        stacks[dst].append(block)
    moving=None
    if completed<7 and index>=4:
        src,dst,block=moves[completed]
        start=position(src,len(stacks[src])-1)
        end=position(dst,len(stacks[dst]))
        stacks[src].pop()
        # All lateral movement occurs above the tallest stack.
        high=min(start[1],end[1],567)-30
        def smooth(t):return t*t*(3-2*t)
        if phase<.28:
            t=smooth(phase/.28);x=start[0];y=start[1]+(high-start[1])*t
        elif phase<.72:
            t=smooth((phase-.28)/.44);x=start[0]+(end[0]-start[0])*t;y=high
        else:
            t=smooth((phase-.72)/.28);x=end[0];y=high+(end[1]-high)*t
        moving=(block,round(x),round(y))
    im=base.copy()
    for stack,blocks in enumerate(stacks):
        for level,block in enumerate(blocks):
            im.paste(sprites[block],position(stack,level),sprites[block])
    if moving:
        block,x,y=moving;im.paste(sprites[block],(x,y),sprites[block])
    if completed:
        draw=ImageDraw.Draw(im)
        draw.rectangle((425,976,599,1013),fill=bg)
        draw.text((512,975),f'Moves: {completed}',font=font,fill=(50,50,50),anchor='mt')
    return im

(ROOT/'output').mkdir(exist_ok=True)
cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','16','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(ROOT/'output/video.mp4')]
process=subprocess.Popen(cmd,stdin=subprocess.PIPE)
for i in range(135):
    im=frame(i)
    process.stdin.write(im.tobytes())
process.stdin.close()
if process.wait()!=0:raise RuntimeError('ffmpeg failed')
assert np.array_equal(np.array(frame(0)),np.array(original))
for i in range(135):
    assert np.array_equal(np.array(frame(i))[50:975,514:],np.array(original)[50:975,514:])
print('Created output/video.mp4: 135 frames, 16 fps; target stays unchanged.')
