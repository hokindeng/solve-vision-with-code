import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
# Two feet, a three-cube rear column, and one forward overhang.
CUBES = {(0,0,0), (1,0,0), (0,1,0), (0,1,1), (0,1,2), (0,0,2)}
FACES = [
 ((1,0,0), [(1,0,0),(1,1,0),(1,1,1),(1,0,1)]),
 ((-1,0,0),[(0,1,0),(0,0,0),(0,0,1),(0,1,1)]),
 ((0,1,0), [(1,1,0),(0,1,0),(0,1,1),(1,1,1)]),
 ((0,-1,0),[(0,0,0),(1,0,0),(1,0,1),(0,0,1)]),
 ((0,0,1), [(0,0,1),(1,0,1),(1,1,1),(0,1,1)]),
 ((0,0,-1),[(0,1,0),(1,1,0),(1,0,0),(0,0,0)])
]

def render(azimuth):
    a, e = math.radians(azimuth), math.radians(36)
    right = np.array([math.cos(a), math.sin(a), 0.])
    up = np.array([-math.sin(e)*math.sin(a), math.sin(e)*math.cos(a), math.cos(e)])
    eye = np.array([math.cos(e)*math.sin(a), -math.cos(e)*math.cos(a), math.sin(e)])
    center = np.array([1.,1.,1.5])
    polygons=[]
    for cube in CUBES:
        for normal, corners in FACES:
            n = np.array(normal)
            if tuple(np.array(cube)+n) in CUBES or np.dot(n,eye) <= 1e-9:
                continue
            points=np.array(corners,dtype=float)+cube
            projected=np.column_stack((512+138*(points-center)@right,512-138*(points-center)@up))
            if normal[2] == 1:
                color=(197,168,225)
            else:
                # Soft, camera-relative illumination, matching the reference's
                # lighter left face and darker right face.
                q = float(n@right)
                t = (q + math.sin(math.radians(10))) / (math.cos(math.radians(10))+math.sin(math.radians(10)))
                color=tuple(int(round(v)) for v in np.array([157,135,180])*(1-t)+np.array([118,101,135])*t)
            polygons.append((float((points.mean(axis=0)-center)@eye),projected,color))
    im=Image.new('RGB',(1024,1024),'white')
    draw=ImageDraw.Draw(im)
    for _, polygon, color in sorted(polygons,key=lambda p:p[0]):
        draw.polygon([tuple(p) for p in polygon],fill=color,outline=(0,0,0),width=1)
    return np.asarray(im)

def main():
    OUT.mkdir(exist_ok=True)
    with imageio.get_writer(OUT/'video.mp4',fps=16,codec='libx264',pixelformat='yuv420p',quality=9,macro_block_size=1) as writer:
        for i in range(21):
            if i == 0:
                frame=np.asarray(Image.open(ROOT/'first_frame.png').convert('RGB'))
            else:
                frame=render(10+180*i/20)
            writer.append_data(frame)

if __name__ == '__main__':
    main()
