from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = Image.open(ROOT / 'first_frame.png').convert('RGB')
arr = np.asarray(base)
background = np.all(arr == 255, axis=2)
S = 3

def curve(points):
    p = np.array(points, dtype=float)
    t = np.linspace(0, 1, 1600)[:, None]
    xy = (1-t)**3*p[0] + 3*(1-t)**2*t*p[1] + 3*(1-t)*t*t*p[2] + t**3*p[3]
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    samples = np.linspace(0, d[-1], 1200)
    return np.column_stack([np.interp(samples, d, xy[:, k]) for k in range(2)])

paths = [
    (curve([(280, 377), (418, 502), (616, 498), (758, 417)]), (170,178,255)),
    (curve([(392, 774), (512, 662), (654, 640), (779, 718)]), (72,219,251)),
]

def frame(progress):
    result = arr.copy()
    for (pts, color), amount in zip(paths, progress):
        if amount <= 0:
            continue
        n = max(2, int(amount * (len(pts)-1)) + 1)
        mask = Image.new('L', (1024*S,1024*S), 0)
        draw = ImageDraw.Draw(mask)
        coords = [tuple(p*S) for p in pts[:n]]
        draw.line(coords, fill=255, width=6*S, joint='curve')
        for x,y in (coords[0],coords[-1]):
            r = 3*S
            draw.ellipse((x-r,y-r,x+r,y+r),fill=255)
        alpha = np.asarray(mask.resize((1024,1024),Image.Resampling.LANCZOS)).astype(float)/255
        alpha *= background
        a = alpha[:,:,None]
        result = np.rint(result*(1-a)+np.array(color)*a).astype(np.uint8)
    return result

cmd = ['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','15','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')]
proc = subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
for i in range(48):
    progress = (min(1,i/23), max(0,min(1,(i-23)/24)))
    img = frame(progress)
    proc.stdin.write(img.tobytes())
proc.stdin.close()
err = proc.stderr.read()
if proc.wait():
    raise RuntimeError(err.decode())
