from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = Image.open(ROOT / 'first_frame.png').convert('RGB')
a = np.asarray(base)
# Cubic curves join only the boundary of each matching pair.
curves = [
    ([(403,288),(470,213),(536,211),(613,262)], tuple(a[303,326])),
    ([(468,559),(529,593),(594,566),(664,525)], tuple(a[540,419])),
    ([(442,795),(499,842),(558,803),(604,745)], tuple(a[750,378])),
]
paths = []
for controls, color in curves:
    p = np.array(controls, dtype=float)
    t = np.linspace(0,1,1201)[:,None]
    xy = (1-t)**3*p[0]+3*(1-t)**2*t*p[1]+3*(1-t)*t*t*p[2]+t**3*p[3]
    length = np.concatenate(([0],np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))))
    paths.append((xy,length/length[-1],color))

cmd = ['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
       '-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264',
       '-preset','slow','-crf','0','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')]
proc = subprocess.Popen(cmd,stdin=subprocess.PIPE)
scale = 3
for frame in range(48):
    canvas = base.copy()
    for k,(xy,length,color) in enumerate(paths):
        progress = np.clip((frame-k*15)/15,0,1)
        if progress <= 0:
            continue
        end = np.searchsorted(length,progress,side='right')
        pts = xy[:max(2,end)]
        mask = Image.new('L',(1024*scale,1024*scale),0)
        draw = ImageDraw.Draw(mask)
        coords = [tuple(v*scale) for v in pts]
        draw.line(coords,fill=255,width=5*scale)
        r = 2.5*scale
        for x,y in (coords[0],coords[-1]):
            draw.ellipse((x-r,y-r,x+r,y+r),fill=255)
        mask = mask.resize(base.size,Image.Resampling.LANCZOS)
        # Original shapes and all untouched background pixels remain intact.
        m = np.asarray(mask).copy()
        m[np.any(a != 255,axis=2)] = 0
        canvas.paste(tuple(map(int,color)),(0,0,1024,1024),Image.fromarray(m))
    proc.stdin.write(canvas.tobytes())
    if frame == 47:
        canvas.save(OUT/'last_frame.png')
proc.stdin.close()
if proc.wait() != 0:
    raise RuntimeError('ffmpeg failed')
