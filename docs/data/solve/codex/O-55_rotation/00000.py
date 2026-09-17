import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT=Path('/app')
OUT=ROOT/'output'
OUT.mkdir(exist_ok=True)
SIZE=1024
# Four cubes in the tower, and two cubes forming the bent foot.
CUBES={(0,0,z) for z in range(4)} | {(0,1,0),(1,1,0)}
TOP=(206,168,206)
SIDE=(123,101,123)
FACES=[((0,0,1),[(0,0,1),(1,0,1),(1,1,1),(0,1,1)]),
       ((1,0,0),[(1,0,0),(1,1,0),(1,1,1),(1,0,1)]),
       ((-1,0,0),[(0,1,0),(0,0,0),(0,0,1),(0,1,1)]),
       ((0,1,0),[(1,1,0),(0,1,0),(0,1,1),(1,1,1)]),
       ((0,-1,0),[(0,0,0),(1,0,0),(1,0,1),(0,0,1)])]

def render(t):
    # The coordinate convention places the given 140-degree view at theta=40.
    theta=math.radians(40+180*t)
    elevation=math.radians(31)
    c,s=math.cos(theta),math.sin(theta)
    ce,se=math.cos(elevation),math.sin(elevation)
    view=np.array([ce*c,ce*s,se])
    right=np.array([s,-c,0.])
    up=np.array([-se*c,-se*s,ce])
    target=np.array([1.,1.,2.])
    scale=103.5
    polys=[]
    for cube in CUBES:
        for normal,verts in FACES:
            if np.dot(normal,view)<=1e-8:
                continue
            neighbor=tuple(cube[i]+normal[i] for i in range(3))
            if neighbor in CUBES:
                continue
            points=np.array(verts,dtype=float)+np.array(cube)
            local=points-target
            screen=np.column_stack((512+scale*(local@right),513.5-scale*(local@up)))
            polys.append((np.mean(points@view),screen,TOP if normal[2] else SIDE))
    im=Image.new('RGB',(SIZE,SIZE),'white')
    draw=ImageDraw.Draw(im)
    for _,screen,color in sorted(polys,key=lambda p:p[0]):
        pts=[tuple(np.rint(p).astype(int)) for p in screen]
        draw.polygon(pts,fill=color)
        draw.line(pts+[pts[0]],fill=(0,0,0),width=1)
    return im

def main():
    command=['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')]
    proc=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    for i in range(21):
        frame=Image.open(ROOT/'first_frame.png').convert('RGB') if i==0 else render(i/20)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    error=proc.stderr.read()
    if proc.wait():
        raise RuntimeError(error.decode())

if __name__=='__main__':
    main()
