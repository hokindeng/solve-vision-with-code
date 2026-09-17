from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = Image.open(ROOT / 'first_frame.png').convert('RGB')
# Trace all eight boundary segments in their original order.
vertices = np.array([(103,452),(435,590),(478,869),(509,735),
                     (666,712),(816,795),(823,867),(888,452)], dtype=float)
lengths = np.linalg.norm(np.roll(vertices,-1,axis=0)-vertices, axis=1)
longest = int(np.argmax(lengths))
midpoint = (vertices[longest]+vertices[(longest+1)%8])/2

# Draw only the temporary edge comparisons and the requested final marker.
# The original image is freshly copied each frame so annotations never accumulate.
def render(index):
    if index == 0:
        return np.asarray(base)
    scale = 4
    overlay = Image.new('RGBA', (1024*scale,1024*scale))
    pen = ImageDraw.Draw(overlay)
    def edge(k, color):
        a, b = vertices[k], vertices[(k+1)%8]
        pen.line([tuple(a*scale),tuple(b*scale)],fill=color,width=2*scale)
    if 1 <= index <= 16:
        # Two frames for each edge, allowing the full boundary to be compared.
        edge((index-1)//2, (224,155,34,255))
    elif index <= 18:
        edge(longest, (224,155,34,255))
    else:
        x,y=midpoint*scale
        r=9*scale
        progress=min(1.0,(index-18)/4)
        pen.arc((x-r,y-r,x+r,y+r),-90,-90+360*progress,
                fill=(230,30,40,255),width=2*scale)
    overlay=overlay.resize(base.size,Image.Resampling.LANCZOS)
    return np.asarray(Image.alpha_composite(base.convert('RGBA'),overlay).convert('RGB'))

command=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo',
         '-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-',
         '-an','-c:v','libx264','-crf','0','-preset','medium',
         '-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')]
process=subprocess.Popen(command,stdin=subprocess.PIPE)
for i in range(25):
    process.stdin.write(render(i).tobytes())
process.stdin.close()
if process.wait():
    raise RuntimeError('ffmpeg failed')
print('Edge lengths:', ', '.join(f'{x:.2f}' for x in lengths))
print('Longest side midpoint:', midpoint.tolist())
