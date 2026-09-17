import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import imageio.v2 as imageio

ROOT=Path('/app')
OUT=ROOT/'output'
OUT.mkdir(exist_ok=True)
# Three cubes across the front, three along the right-hand return
# (sharing the corner), and one cube above the middle of the return.
CUBES={(0,0,0),(1,0,0),(2,0,0),(2,1,0),(2,2,0),(2,1,1)}
ELEV=math.radians(28)
SCALE=138.0
TARGET=np.array([1.5,1.5,1.0])
A0=math.radians(20)
# Keep the orbit target fixed at the projection implied by the reference.
CENTER=np.array([247+1.5*SCALE*(math.cos(A0)+math.sin(A0)),
                 693+1.5*SCALE*math.sin(ELEV)*(math.sin(A0)-math.cos(A0))-SCALE*math.cos(ELEV)])
FACES=[((1,0,0),[(1,0,0),(1,1,0),(1,1,1),(1,0,1)],(135,118,95)),
       ((-1,0,0),[(0,1,0),(0,0,0),(0,0,1),(0,1,1)],(180,157,127)),
       ((0,-1,0),[(0,0,0),(1,0,0),(1,0,1),(0,0,1)],(180,157,127)),
       ((0,1,0),[(1,1,0),(0,1,0),(0,1,1),(1,1,1)],(135,118,95)),
       ((0,0,1),[(0,0,1),(1,0,1),(1,1,1),(0,1,1)],(225,197,159))]

def render(azimuth):
    a=math.radians(azimuth)
    right=np.array([math.cos(a),math.sin(a),0])
    up=np.array([-math.sin(ELEV)*math.sin(a), math.sin(ELEV)*math.cos(a),math.cos(ELEV)])
    view=np.array([math.cos(ELEV)*math.sin(a),-math.cos(ELEV)*math.cos(a),math.sin(ELEV)])
    faces=[]
    for cube in CUBES:
        for normal,verts,color in FACES:
            neighbor=tuple(cube[i]+normal[i] for i in range(3))
            if neighbor in CUBES or np.dot(normal,view)<=1e-8:
                continue
            world=np.array(verts,dtype=float)+np.array(cube)
            relative=world-TARGET
            points=np.column_stack((relative@right,-relative@up))*SCALE+CENTER
            faces.append((world.mean(axis=0)@view,points,color))
    im=Image.new('RGB',(1024,1024),'white')
    draw=ImageDraw.Draw(im)
    for _,points,color in sorted(faces,key=lambda x:x[0]):
        polygon=[tuple(np.rint(p).astype(int)) for p in points]
        draw.polygon(polygon,fill=color,outline=(0,0,0),width=1)
    return im

def main():
    first=Image.open(ROOT/'first_frame.png').convert('RGB')
    with imageio.get_writer(OUT/'video.mp4',format='FFMPEG',mode='I',fps=16,
                           codec='libx264',pixelformat='yuv420p',macro_block_size=1,
                           ffmpeg_params=['-crf','18','-preset','slow']) as writer:
        for i in range(21):
            frame=first if i==0 else render(20+180*i/20)
            writer.append_data(np.asarray(frame))
    render(200).save(OUT/'last_frame.png')

if __name__=='__main__':
    main()
