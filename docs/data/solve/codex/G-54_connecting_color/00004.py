from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = Image.open(ROOT / 'first_frame.png').convert('RGB')
a = np.array(base)
# Use the original solid fill colors for both connecting strokes.
orange = tuple(int(v) for v in a[200,123])
pink = tuple(int(v) for v in a[600,330])
curves = [
    (orange, [(175,200),(340,155),(465,339),(630,322)]),
    (pink, [(362,582),(530,536),(658,756),(836,715)]),
]
S = 4

def sampled_curve(points):
    t = np.linspace(0,1,1601)[:,None]
    p = np.array(points,dtype=float)
    return (1-t)**3*p[0] + 3*(1-t)**2*t*p[1] + 3*(1-t)*t*t*p[2] + t**3*p[3]

paths=[]
for color, points in curves:
    xy=sampled_curve(points)
    length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
    paths.append((color,xy,length/length[-1]))

# Preserve all original shape pixels, including their antialiased edges.
shapes = np.any(a != 255,axis=2)
command=['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24',
         '-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264',
         '-preset','slow','-crf','12','-pix_fmt','yuv420p','-movflags','+faststart',
         str(OUT/'video.mp4')]
proc=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
for frame in range(48):
    layer=Image.new('RGBA',(1024*S,1024*S),(0,0,0,0))
    draw=ImageDraw.Draw(layer)
    for k,(color,xy,dist) in enumerate(paths):
        progress=np.clip((frame-(0 if k==0 else 23))/23,0,1)
        if progress <= 0:
            continue
        n=np.searchsorted(dist,progress,side='right')
        pts=[tuple(p*S) for p in xy[:max(2,n)]]
        draw.line(pts,fill=color+(255,),width=6*S,joint='curve')
        for x,y in (pts[0],pts[-1]):
            r=3*S
            draw.ellipse((x-r,y-r,x+r,y+r),fill=color+(255,))
    layer=layer.resize(base.size,Image.Resampling.LANCZOS)
    result=np.array(Image.alpha_composite(base.convert('RGBA'),layer).convert('RGB'))
    result[shapes]=a[shapes]
    proc.stdin.write(result.tobytes())
proc.stdin.close()
if proc.wait()!=0:
    raise RuntimeError('ffmpeg failed')
