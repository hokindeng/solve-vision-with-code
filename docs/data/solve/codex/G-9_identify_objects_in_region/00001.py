from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
# Only the two trapezoids inside the right-hand square are selected.
POLYGONS = [ [(602,370),(626,370),(634,410),(594,410)],
             [(745,405),(783,405),(797,471),(731,471)] ]

def trace(draw, vertices, progress):
    points = vertices + [vertices[0]]
    lengths = [float(np.hypot(b[0]-a[0],b[1]-a[1])) for a,b in zip(points,points[1:])]
    remaining = sum(lengths) * min(1., max(0.,progress))
    for a,b,length in zip(points,points[1:],lengths):
        if remaining <= 0:
            break
        fraction = min(1.,remaining/length)
        end = (round(a[0]+(b[0]-a[0])*fraction), round(a[1]+(b[1]-a[1])*fraction))
        draw.line([a,end], fill=(0,170,0), width=3)
        remaining -= length

base = Image.open(ROOT / 'first_frame.png').convert('RGB')
command = ['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24',
           '-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264',
           '-preset','slow','-crf','0','-pix_fmt','yuv420p','-movflags','+faststart',
           str(OUT/'video.mp4')]
proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
for frame in range(40):
    image = base.copy()
    draw = ImageDraw.Draw(image)
    # Draw each perimeter in sequence throughout the 2.5-second clip.
    trace(draw,POLYGONS[0],frame/19)
    trace(draw,POLYGONS[1],(frame-19)/20)
    proc.stdin.write(image.tobytes())
proc.stdin.close()
error = proc.stderr.read()
if proc.wait():
    raise RuntimeError(error.decode())
