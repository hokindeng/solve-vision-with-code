from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

BASE = '/app'
os.makedirs(BASE + '/output', exist_ok=True)
original = Image.open(BASE + '/first_frame.png').convert('RGB')
S = 3
font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font = ImageFont.truetype(font_path, 19*S)
red = (220, 38, 38, 255)

def label(draw, x, y, text):
    draw.text((x*S, y*S), text, font=font, fill=red, anchor='mm')

def arc(draw, box, end, width=3):
    draw.arc(tuple(int(v*S) for v in box), start=-90, end=-90+end, fill=red, width=width*S)

import subprocess
p = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',BASE+'/output/video.mp4'], stdin=subprocess.PIPE)
for i in range(60):
    layer = Image.new('RGBA', (1024*S,1024*S))
    d = ImageDraw.Draw(layer)
    # Read the three given sizes in order, with a short pause at each.
    steps = [(5,241,337,29,'SMALL'), (15,421,337,67,'LARGE'), (25,602,337,29,'SMALL')]
    for start,x,y,r,word in steps:
        if i >= start:
            label(d,x,435,word)
            if i < start+9:
                arc(d,(x-r,y-r,x+r,y+r),min(360,(i-start+1)*45),2)
    if i >= 35:
        label(d,782,435,'LARGE')
    if i >= 39:
        # Place the inferred figure inside the existing dashed answer box.
        d.rectangle((771*S,323*S,791*S,349*S), fill=(255,255,255,255))
        pts=[(755,283),(809,283),(836,338),(809,392),(755,392),(728,338)]
        d.polygon([(x*S,y*S) for x,y in pts],fill=(35,35,35,255))
    if i >= 44:
        arc(d,(778,776,935,932),min(360,(i-43)*30),5)
    overlay = layer.resize(original.size,Image.Resampling.LANCZOS)
    frame = Image.alpha_composite(original.convert('RGBA'), overlay).convert('RGB')
    p.stdin.write(np.asarray(frame).tobytes())
p.stdin.close()
if p.wait() != 0:
    raise RuntimeError('ffmpeg failed')
