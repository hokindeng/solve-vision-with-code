from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
BLOCKS = {(0,0,0), (0,0,1), (0,1,1), (0,2,1), (0,3,1), (1,3,1)}
# The reference is an orthographic view of unit cubes.
SCALE = 103.5
ELEVATION = math.radians(25)
TARGET = np.array([1., 2., 1.])
FACES = [
    ((-1,0,0), [(0,0,0),(0,1,0),(0,1,1),(0,0,1)], (165,120,120)),
    ((1,0,0), [(1,0,0),(1,1,0),(1,1,1),(1,0,1)], (165,120,120)),
    ((0,-1,0), [(0,0,0),(1,0,0),(1,0,1),(0,0,1)], (123,90,90)),
    ((0,1,0), [(0,1,0),(1,1,0),(1,1,1),(0,1,1)], (123,90,90)),
    ((0,0,1), [(0,0,1),(1,0,1),(1,1,1),(0,1,1)], (206,150,150)),
    ((0,0,-1), [(0,0,0),(1,0,0),(1,1,0),(0,1,0)], (123,90,90)),
]

def render(t):
    theta = math.radians(110 + 180*t)
    direction = np.array([math.cos(ELEVATION)*math.cos(theta), math.cos(ELEVATION)*math.sin(theta), math.sin(ELEVATION)])
    right = np.array([math.sin(theta), -math.cos(theta), 0.])
    down = np.array([math.sin(ELEVATION)*math.cos(theta), math.sin(ELEVATION)*math.sin(theta), -math.cos(ELEVATION)])
    polygons = []
    for block in sorted(BLOCKS):
        base = np.array(block)
        for normal, corners, color in FACES:
            if tuple(base + normal) in BLOCKS or np.dot(normal, direction) <= 1e-9:
                continue
            pts = np.array(corners) + base
            relative = pts - TARGET
            screen = np.column_stack((512 + SCALE*(relative @ right), 512 + SCALE*(relative @ down)))
            polygons.append((float((pts.mean(axis=0)-TARGET) @ direction), screen, color))
    image = Image.new('RGB', (1024,1024), 'white')
    draw = ImageDraw.Draw(image)
    for _, polygon, color in sorted(polygons, key=lambda p:p[0]):
        points = [tuple(np.rint(p).astype(int)) for p in polygon]
        draw.polygon(points, fill=color, outline=(0,0,0), width=1)
    return image

def main():
    first = Image.open(ROOT/'first_frame.png').convert('RGB')
    with imageio.get_writer(OUT/'video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=1, ffmpeg_params=['-crf','18']) as writer:
        for i in range(21):
            writer.append_data(np.asarray(first if i == 0 else render(i/20)))

if __name__ == '__main__':
    main()
