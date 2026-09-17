#!/usr/bin/env python3
"""Regenerate /app/output/video.mp4: camera orbits a fixed 7-block sculpture from
azimuth 20 deg to 200 deg at a constant 32 deg elevation (orthographic camera)."""
import os, subprocess, tempfile
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output', 'video.mp4')
FIRST = os.path.join(HERE, 'first_frame.png')

SIZE = 1024
FPS = 16
NFRAMES = 21
AZ0, AZ1 = 20.0, 200.0
EL = 32.0

# Scene (unit cubes on the z=0 table plane) and camera parameters fitted to first_frame.png
BLOCKS = [(0,0,0),(0,1,0),(0,2,0),(-1,0,0),(-2,0,0),(-2,0,1),(-2,0,2)]
SCALE, CX, CY = 137.9, 512.46, 512.43          # orthographic pixels per unit, image centre
BASE = np.array([215.0, 205.7, 168.4])          # block colour before shading
AMBIENT = 0.4
# Light direction expressed in camera coordinates (right, up, toward camera); it rides with
# the camera so faces keep the same 1.0 / 0.8 / 0.6 shading pattern seen in the first frame.
LIGHT_CAM = None  # filled in below from the world light that reproduces frame 0


def cam_basis(az, el):
    az, el = np.radians(az), np.radians(el)
    fwd = -np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    right = np.array([-np.sin(az), np.cos(az), 0.0])
    up = np.cross(right, fwd)
    return fwd, right, up


def _init_light():
    global LIGHT_CAM
    Lw = np.array([0.4, 0.2, 0.6])   # n.L = 0.6 (+z), 0.4 (+x), 0.2 (+y) at frame 0
    fwd, right, up = cam_basis(AZ0, EL)
    LIGHT_CAM = np.array([Lw @ right, Lw @ up, Lw @ (-fwd)])
_init_light()


def faces_of_block(b):
    x, y, z = b
    v = lambda dx, dy, dz: np.array([x+dx, y+dy, z+dz], float)
    return [
        ([v(1,0,0), v(1,1,0), v(1,1,1), v(1,0,1)], np.array([1, 0, 0.])),
        ([v(0,1,0), v(0,0,0), v(0,0,1), v(0,1,1)], np.array([-1, 0, 0.])),
        ([v(1,1,0), v(0,1,0), v(0,1,1), v(1,1,1)], np.array([0, 1, 0.])),
        ([v(0,0,0), v(1,0,0), v(1,0,1), v(0,0,1)], np.array([0, -1, 0.])),
        ([v(0,0,1), v(1,0,1), v(1,1,1), v(0,1,1)], np.array([0, 0, 1.])),
        ([v(0,1,0), v(1,1,0), v(1,0,0), v(0,0,0)], np.array([0, 0, -1.])),
    ]


def render(az, el):
    blocks_set = set(BLOCKS)
    arr = np.array(BLOCKS, float)
    target = (arr.min(0) + arr.max(0) + 1) / 2
    fwd, right, up = cam_basis(az, el)
    Lw = LIGHT_CAM[0]*right + LIGHT_CAM[1]*up + LIGHT_CAM[2]*(-fwd)
    img = Image.new('RGB', (SIZE, SIZE), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    polys = []
    for b in BLOCKS:
        for verts, n in faces_of_block(b):
            nb = tuple(int(round(c)) for c in (np.array(b) + n))
            if nb in blocks_set or np.dot(n, fwd) >= 0:
                continue  # hidden interior face or back face
            depth = np.mean([np.dot(v - target, fwd) for v in verts])
            pts = [(CX + SCALE*np.dot(v - target, right), CY - SCALE*np.dot(v - target, up))
                   for v in verts]
            inten = AMBIENT + max(0.0, float(np.dot(n, Lw)))
            col = tuple(int(round(c)) for c in np.clip(BASE*inten, 0, 255))
            polys.append((depth, pts, col))
    polys.sort(key=lambda p: p[0], reverse=True)  # painter's algorithm, far faces first
    for _, pts, col in polys:
        dr.polygon(pts, fill=col, outline=(0, 0, 0))
    return img


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        for i in range(NFRAMES):
            t = i / (NFRAMES - 1)
            az = AZ0 + (AZ1 - AZ0) * t
            im = render(az, EL)
            if i == 0 and os.path.exists(FIRST):
                ref = Image.open(FIRST).convert('RGB')
                diff = np.abs(np.asarray(im, int) - np.asarray(ref, int)).max()
                if diff > 3:
                    print('warning: rendered frame 0 deviates from first_frame.png by', diff)
                im = ref  # first frame is exactly the given image
            im.save(os.path.join(td, f'f{i:03d}.png'))
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS),
                        '-i', os.path.join(td, 'f%03d.png'),
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '12', '-preset', 'slow',
                        '-movflags', '+faststart', OUT], check=True)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
