from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
BASE = Image.open(ROOT / 'first_frame.png').convert('RGB')
S = 3
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

def frame(i):
    if i == 0:
        return np.array(BASE)
    layer = Image.new('RGBA', (1024*S, 1024*S))
    d = ImageDraw.Draw(layer)
    gray = (90,90,90,255)
    red = (222,20,51,255)
    def line(points, fill=gray, width=2):
        d.line([(int(x*S),int(y*S)) for x,y in points], fill=fill, width=width*S)
    def text(x,y,t,size=21,fill=gray):
        d.text((x*S,y*S),t,font=ImageFont.truetype(FONT,size*S),fill=fill,anchor='mm')
    def arrow(x1,x2,y,p):
        x = x1+(x2-x1)*p
        line([(x1,y),(x,y)])
        if p >= 1:
            line([(x-8,y-5),(x,y),(x-8,y+5)])
    # Compare the first two diameter increments, one at a time.
    if i >= 5:
        text(512,235,'Same orange circles · increasing size',23)
        for x,r in [(241,10),(421,19)]:
            line([(x-r,387),(x+r,387)])
            line([(x-r,383),(x-r,391)])
            line([(x+r,383),(x+r,391)])
        arrow(271,384,415,min(1,(i-5)/8))
        if i >= 13:
            text(330,443,'+1 size step',18)
    if i >= 18:
        line([(574,387),(630,387)])
        line([(574,383),(574,391)])
        line([(630,383),(630,391)])
        arrow(452,563,415,min(1,(i-18)/8))
        if i >= 26:
            text(511,443,'+1 size step',18)
    if i >= 31:
        arrow(643,740,415,min(1,(i-31)/7))
        if i >= 38:
            text(692,443,'+1 size step',18)
    if i >= 39:
        # Reveal the predicted next circle inside the existing placeholder.
        alpha = round(255*min(1,(i-38)/6))
        d.ellipse((746*S,301*S,820*S,375*S),fill=(255,140,0,alpha))
    if i >= 45:
        text(512,545,'Choose the matching next size',23)
    if i >= 47:
        p = min(1,(i-47)/9)
        # Draw a red ring around the matching option over nine frames.
        points=[]
        for k in range(int(180*p)+1):
            a=-math.pi/2+2*math.pi*k/180
            points.append((396+62*math.cos(a),864+62*math.sin(a)))
        if len(points)>1:
            line(points,red,5)
    if i >= 56:
        text(512,589,'Second option: same shape, color, and size step',21,red)
    overlay = layer.resize(BASE.size,Image.Resampling.LANCZOS)
    return np.array(Image.alpha_composite(BASE.convert('RGBA'),overlay).convert('RGB'))

# Encode lossless H.264 so the source artwork remains stable throughout.
import subprocess
cmd=['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-preset','medium','-crf','0','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')]
p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
for i in range(60):
    p.stdin.write(frame(i).tobytes())
p.stdin.close()
err=p.stderr.read()
if p.wait():
    raise RuntimeError(err.decode())
