from pathlib import Path
import math
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
BASE = Image.open(ROOT / 'first_frame.png').convert('RGB')
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
S = 3

def frame(i):
    if i == 0:
        return BASE.copy()
    layer = Image.new('RGBA', (1024*S, 1024*S))
    d = ImageDraw.Draw(layer)
    gray = (85,85,85,255)
    red = (220,30,35,255)
    def line(points, color=gray, width=2):
        d.line([(int(x*S),int(y*S)) for x,y in points], fill=color, width=width*S)
    def text(x,y,value,size=22,color=gray):
        d.text((x*S,y*S),value,font=ImageFont.truetype(FONT,size*S),fill=color,anchor='mm')
    def arrow(x1,x2,y):
        line([(x1,y),(x2,y)])
        line([(x2-7,y-5),(x2,y),(x2-7,y+5)])
    # Read the first pair, then repeat the same measurement for the second.
    if i >= 5:
        text(241,425,'47 px')
    if i >= 10:
        text(421,425,'67 px')
        arrow(284,378,465)
        text(331,491,'+20 px',20)
    if i >= 20:
        text(602,425,'87 px')
        arrow(465,558,465)
        text(512,491,'+20 px',20)
    if i >= 30:
        arrow(646,739,465)
        text(692,491,'+20 px',20)
        text(782,425,'107 px')
    if i >= 37:
        # The predicted next triangle has exactly the size of choice three.
        # Paste only its colored pixels; preserve the dashed target border.
        a = np.array(BASE)
        crop = a[811:918,573:680]
        mask = np.all(crop == (165,42,42), axis=2)
        tile = Image.fromarray(crop).convert('RGBA')
        tile.putalpha(Image.fromarray((mask*255).astype('uint8')))
        layer.alpha_composite(tile.resize((107*S,107*S), Image.Resampling.NEAREST),(729*S,284*S))
    if i >= 43:
        progress = min(1,(i-42)/12)
        d.arc((543*S,783*S,710*S,946*S),start=-90,end=-90+360*progress,fill=red,width=4*S)
    overlay = layer.resize(BASE.size,Image.Resampling.LANCZOS)
    return Image.alpha_composite(BASE.convert('RGBA'),overlay).convert('RGB')

proc = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size','1024x1024','-framerate','16','-i','-','-an','-c:v','libx264','-preset','slow','-crf','0','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')],stdin=subprocess.PIPE)
for i in range(60):
    proc.stdin.write(np.asarray(frame(i)).tobytes())
proc.stdin.close()
if proc.wait() != 0:
    raise RuntimeError('ffmpeg failed')
