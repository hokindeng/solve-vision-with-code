from pathlib import Path
import math
import cv2
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
source = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
# The original artwork has flat, pixel-exact disk silhouettes.
black_mask = np.all(source == (0, 0, 0), axis=2)
base = source.copy()
base[black_mask] = 255
Y, X = np.mgrid[:1024, :1024]
colors = [(255,215,0), (255,140,0), (70,130,180), (220,20,60), (255,215,0)]
centers = [(583,313), (153,382), (120,918), (754,540), (830,142)]
masks = []
for color, (cx,cy) in zip(colors, centers):
    mask = np.all(source == color, axis=2)
    # Separate the two yellow disks without changing their pixels.
    mask &= (X-cx)**2 + (Y-cy)**2 < 105**2
    masks.append(mask)
positions = [(651,371), (583,313), (153,382), (140,880), (754,540), (815,190)]
radii = [math.sqrt(5401/math.pi), 65, 90, 114, 145, 175]

def ease(t):
    t = np.clip(t,0,1)
    return t*t*(3-2*t)

def frame(i):
    if i <= 2:
        return source.copy()
    step = min((i-3)//20,4)
    t = min(((i-3)%20+1)/20,1) if i < 103 else 1.0
    picture = base.copy()
    for j in range(step):
        picture[masks[j]] = 255
    a = np.array(positions[step],dtype=float)
    b = np.array(positions[step+1],dtype=float)
    # Travel to each smaller ball, then grow as it is engulfed.
    p = a + (b-a)*ease(t/0.82)
    r = radii[step] + (radii[step+1]-radii[step])*ease((t-0.72)/0.28)
    if t >= 1:
        picture[masks[step]] = 255
    picture[(X-p[0])**2 + (Y-p[1])**2 <= r*r] = 0
    return picture

proc = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','15','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')],stdin=subprocess.PIPE)
for i in range(108):
    proc.stdin.write(frame(i).tobytes())
proc.stdin.close()
if proc.wait():
    raise RuntimeError('ffmpeg failed')
