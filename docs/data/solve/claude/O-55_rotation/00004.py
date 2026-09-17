#!/usr/bin/env python3
"""Regenerate /app/output/video.mp4: camera orbits 180 deg (azimuth 20 -> 200) around a
fixed 6-block sculpture while keeping a 28 deg elevation.  The scene is reconstructed from
first_frame.png (flat-shaded cubes, 1px black outlines, white background, orthographic view
looking at the bounding-box centre of the sculpture).
"""
import os, subprocess, shutil
import numpy as np
from PIL import Image, ImageDraw

W = H = 1024
FPS = 16
N_FRAMES = 21
AZ0, AZ1 = 20.0, 200.0
EL = 28.0

# 6 unit cubes: (ix, iy, iz) -> cube occupies [ix-.5, ix+.5] x [iy-.5, iy+.5] x [iz, iz+1]
CUBES = [(0, 0, 0), (0, 1, 0), (0, 2, 0), (-1, 2, 0), (-2, 2, 0), (-1, 2, 1)]
BASE = np.array([225, 197, 159], dtype=float)
EDGE = (0, 0, 0)
BG = (255, 255, 255)

# Flat shading per face normal. Top/+x/+y (1.0/0.8/0.6, truncated to int) reproduce the
# first frame exactly; the -x/-y faces are only revealed during the orbit and are given
# plausible darker values (they face away from the implied light).
FACE_SHADE = {(0, 0, 1): 1.0, (1, 0, 0): 0.8, (0, 1, 0): 0.6,
              (-1, 0, 0): 0.5, (0, -1, 0): 0.4, (0, 0, -1): 0.3}

# Camera parameters (fit to first_frame.png, see fit_camera.py)
CAM = dict(scale=137.9, cx=512.5, cy=512.6, invd=0.0)  # invd = 1/distance; 0 -> orthographic


def bbox_center():
    c = np.array(CUBES, dtype=float)
    lo = c.min(0) + np.array([-0.5, -0.5, 0.0]); hi = c.max(0) + np.array([0.5, 0.5, 1.0])
    return (lo + hi) / 2


def camera(az_deg, el_deg):
    az, el = np.radians(az_deg), np.radians(el_deg)
    v = np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])   # toward camera
    right = np.array([-np.sin(az), np.cos(az), 0.0])
    up = np.cross(v, right)  # (-cos az sin el, -sin az sin el, cos el)
    return v, right, up


def project(P, az, el, cam):
    v, right, up = camera(az, el)
    T = bbox_center()
    rel = P - T
    x = rel @ right; y = rel @ up; depth = rel @ v  # depth: + toward camera
    invd = cam['invd']
    if invd > 0:
        d = 1.0 / invd
        k = d / (d - depth)
    else:
        k = 1.0
    sx = cam['cx'] + cam['scale'] * x * k
    sy = cam['cy'] - cam['scale'] * y * k
    return np.stack([sx, sy], -1), depth


FACES = {  # normal -> corner offsets (ccw seen from outside), relative to cube centre
    (1, 0, 0): [(.5, -.5, -.5), (.5, .5, -.5), (.5, .5, .5), (.5, -.5, .5)],
    (-1, 0, 0): [(-.5, .5, -.5), (-.5, -.5, -.5), (-.5, -.5, .5), (-.5, .5, .5)],
    (0, 1, 0): [(.5, .5, -.5), (-.5, .5, -.5), (-.5, .5, .5), (.5, .5, .5)],
    (0, -1, 0): [(-.5, -.5, -.5), (.5, -.5, -.5), (.5, -.5, .5), (-.5, -.5, .5)],
    (0, 0, 1): [(-.5, -.5, .5), (.5, -.5, .5), (.5, .5, .5), (-.5, .5, .5)],
    (0, 0, -1): [(-.5, .5, -.5), (.5, .5, -.5), (.5, -.5, -.5), (-.5, -.5, -.5)],
}


def shade(n):
    f = FACE_SHADE[tuple(int(round(c)) for c in n)]
    return tuple(int(c * f) for c in BASE)


def render(az, el=EL, cam=CAM, edges_only=False):
    img = Image.new('RGB', (W, H), BG)
    dr = ImageDraw.Draw(img)
    v, right, up = camera(az, el)
    T = bbox_center()
    centers = [np.array([ix, iy, iz + 0.5], dtype=float) for ix, iy, iz in CUBES]
    invd = cam['invd']
    if invd > 0:
        C = T + v / invd
        order = sorted(range(len(centers)), key=lambda i: -np.linalg.norm(centers[i] - C))
    else:
        order = sorted(range(len(centers)), key=lambda i: centers[i] @ v)
    for i in order:
        c = centers[i]
        for n, offs in FACES.items():
            n = np.array(n, dtype=float)
            fc = c + 0.5 * n
            vis = (np.dot(n, C - fc) > 0) if invd > 0 else (np.dot(n, v) > 0)
            if not vis:
                continue
            P = c + np.array(offs)
            pts, _ = project(P, az, el, cam)
            poly = [tuple(p) for p in pts]
            if edges_only:
                dr.polygon(poly, fill=None, outline=EDGE)
            else:
                dr.polygon(poly, fill=shade(n), outline=EDGE)
    return img


def main():
    out_dir = '/app/output'
    os.makedirs(out_dir, exist_ok=True)
    frames_dir = os.path.join(out_dir, 'frames')
    shutil.rmtree(frames_dir, ignore_errors=True)
    os.makedirs(frames_dir)
    first = Image.open('/app/first_frame.png').convert('RGB')
    for k in range(N_FRAMES):
        t = k / (N_FRAMES - 1)
        az = AZ0 + (AZ1 - AZ0) * t
        img = render(az)
        if k == 0:
            diff = np.abs(np.asarray(img, dtype=int) - np.asarray(first, dtype=int))
            print('frame0 vs first_frame: max diff', diff.max(), 'pixels differing', int((diff.sum(2) > 0).sum()))
            img = first  # guarantee an exact first frame
        img.save(os.path.join(frames_dir, f'{k:03d}.png'))
    mp4 = os.path.join(out_dir, 'video.mp4')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS),
                    '-i', os.path.join(frames_dir, '%03d.png'),
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '12', '-preset', 'slow',
                    '-r', str(FPS), mp4], check=True)
    print('wrote', mp4)


if __name__ == '__main__':
    main()
