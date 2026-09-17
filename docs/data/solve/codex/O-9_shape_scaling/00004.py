from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = Image.open(ROOT / 'first_frame.png').convert('RGB')
# The example hexagons span 150 and 130 pixels between their vertices.
SCALE = 130 / 150
N = 60

def smooth(t):
    t = min(1., max(0., t))
    return t*t*(3-2*t)

def frame(i):
    if i == 0:
        return base.copy()
    im = base.copy()
    # Only the answer cell changes. Fade the existing question mark away.
    box = (690, 685, 850, 850)
    patch = base.crop(box)
    erase = smooth(i / 15)
    im.paste(Image.blend(patch, Image.new('RGB', patch.size, 'white'), erase), box)
    if i >= 15:
        # Show the trapezoid at its original size, then smoothly apply A:B's scale.
        alpha = smooth((i-15)/12)
        scale = 1 + (SCALE-1)*smooth((i-22)/33)
        layer = Image.new('RGB', (160,165), 'white')
        draw = ImageDraw.Draw(layer)
        pts = [(769 + x*scale - box[0], 768.5 + y*scale - box[1])
               for x,y in [(-37.5,-75),(37.5,-75),(75,75),(-75,75)]]
        draw.polygon(pts, fill=(191,0,64), outline=(0,0,0))
        im.paste(Image.blend(Image.new('RGB', layer.size, 'white'),layer,alpha),box)
    return im

proc = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo',
    '-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16',
    '-i','-','-an','-c:v','libx264','-preset','slow','-crf','0',
    '-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')], stdin=subprocess.PIPE)
for i in range(N):
    proc.stdin.write(np.asarray(frame(i)).tobytes())
proc.stdin.close()
if proc.wait():
    raise RuntimeError('ffmpeg failed')
