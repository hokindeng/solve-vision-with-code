import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT=Path('/app')
OUT=ROOT/'output'
OUT.mkdir(exist_ok=True)
# Three cubes along the front, three along the return, and a
# three-cube tower sharing the last cube of the return: seven total.
cubes={(0,0,0),(1,0,0),(2,0,0),(0,1,0),(0,2,0),(0,2,1),(0,2,2)}
faces=[((1,0,0),[(1,0,0),(1,1,0),(1,1,1),(1,0,1)]),
       ((-1,0,0),[(0,1,0),(0,0,0),(0,0,1),(0,1,1)]),
       ((0,1,0),[(1,1,0),(0,1,0),(0,1,1),(1,1,1)]),
       ((0,-1,0),[(0,0,0),(1,0,0),(1,0,1),(0,0,1)]),
       ((0,0,1),[(0,0,1),(1,0,1),(1,1,1),(0,1,1)]),
       ((0,0,-1),[(0,1,0),(1,1,0),(1,0,0),(0,0,0)])]
colors={(1,0,0):(129,123,101),(-1,0,0):(172,165,135),
        (0,-1,0):(172,165,135),(0,1,0):(129,123,101),
        (0,0,1):(215,206,168),(0,0,-1):(110,105,86)}

def render(angle):
    a=math.radians(angle); e=math.radians(32)
    right=np.array([math.cos(a),math.sin(a),0.])
    down=np.array([math.sin(e)*math.sin(a),-math.sin(e)*math.cos(a),-math.cos(e)])
    eye=np.array([math.cos(e)*math.sin(a),-math.cos(e)*math.cos(a),math.sin(e)])
    scale=138.35
    # Keep the same framing while orbiting the sculpture's center.
    center=np.array([1.5,1.5,1.5])
    origin=np.array([512.8,512.5])
    polygons=[]
    for cube in cubes:
        for normal,vertices in faces:
            n=np.array(normal)
            if tuple(np.array(cube)+n) in cubes or np.dot(n,eye)<=1e-8:
                continue
            world=np.array(vertices,dtype=float)+cube
            points=world-center
            screen=np.column_stack((points@right,points@down))*scale+origin
            polygons.append((np.mean(points@eye),screen,colors[normal]))
    im=Image.new('RGB',(1024,1024),'white')
    draw=ImageDraw.Draw(im)
    for _,screen,color in sorted(polygons,key=lambda p:p[0]):
        pts=[tuple(np.rint(p).astype(int)) for p in screen]
        draw.polygon(pts, fill=color)
        draw.line(pts+[pts[0]],fill=(0,0,0),width=1)
    return im

frames=[]
for i in range(21):
    if i==0:
        im=Image.open(ROOT/'first_frame.png').convert('RGB')
    else:
        im=render(20+180*i/20)
    frames.append(im)
proc=subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','16','-preset','slow','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
_,err=proc.communicate(b''.join(im.tobytes() for im in frames))
if proc.returncode:
    raise RuntimeError(err.decode())
frames[-1].save(OUT/'last_frame.png')
