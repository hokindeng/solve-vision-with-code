#!/usr/bin/env python3
"""Render a camera orbit (azimuth 70 -> 250 deg, elevation 25 deg) around a
fixed 6-block sculpture, reproducing /app/first_frame.png as the first frame.

Scene reconstruction (fitted to first_frame.png):
  * orthographic projection, ~103.46 px per block unit
  * camera looks at the sculpture's bounding-box centre (0, 2, 1)
  * blocks occupy unit cells [x,x+1]x[y,y+1]x[z,z+1]
  * flat shading, 1px black outlines, no anti-aliasing, white background
"""
import math
import os
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw

W = H = 1024
FPS = 16
N_FRAMES = 21
AZ0, AZ1 = 70.0, 250.0
ELEV = 25.0
SCALE = 103.46            # pixels per block unit
BG = (255, 255, 255)
EDGE = (0, 0, 0)

# cubes as (x, y, z) of their min corner
CUBES = [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 2, 1), (0, 3, 1), (-1, 3, 1)]

# camera target = bounding-box centre of the sculpture
_arr = np.array(CUBES, dtype=float)
TARGET = (_arr.min(0) + _arr.max(0) + 1.0) / 2.0   # (0, 2, 1)

# Flat shading: base colour * (ambient + diffuse * max(0, n.l)), light fixed
# in the world. Calibrated so +z/+y/+x faces give exactly the first-frame
# colours (206,150,150) / (123,90,90) / (165,120,120).
BASE = np.array([220.0, 160.0, 160.0])
LIGHT = np.array([2.0, 1.0, 3.0]) / math.sqrt(14.0)
AMBIENT, DIFFUSE = 0.375, 0.7026

FACES = {
    # normal : list of 4 corner offsets (counter-clockwise seen from outside)
    (0, 0, 1): [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
    (0, 0, -1): [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
    (1, 0, 0): [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],
    (-1, 0, 0): [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],
    (0, 1, 0): [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)],
    (0, -1, 0): [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
}


def face_color(normal):
    ndotl = max(0.0, float(np.dot(normal, LIGHT)))
    f = AMBIENT + DIFFUSE * ndotl
    return tuple(int(c) for c in BASE * f)


FACE_COLORS = {n: face_color(np.array(n, dtype=float)) for n in FACES}


def camera_basis(az_deg, el_deg):
    az, el = math.radians(az_deg), math.radians(el_deg)
    d = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])  # towards camera
    r = np.array([-math.sin(az), math.cos(az), 0.0])                                        # screen right
    u = np.cross(d, r)                                                                       # screen up
    return d, r, u


def render(az_deg, el_deg=ELEV):
    d, r, u = camera_basis(az_deg, el_deg)
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    def project(p):
        q = np.asarray(p, dtype=float) - TARGET
        return (W / 2.0 + SCALE * float(q @ r), H / 2.0 - SCALE * float(q @ u))

    # painter's algorithm: far cubes first
    order = sorted(CUBES, key=lambda c: float((np.array(c) + 0.5) @ d))
    for cx, cy, cz in order:
        for normal, corners in FACES.items():
            if float(np.dot(normal, d)) <= 0:
                continue  # back-facing
            pts = [project((cx + ox, cy + oy, cz + oz)) for ox, oy, oz in corners]
            draw.polygon(pts, fill=FACE_COLORS[normal], outline=EDGE)
    return img


def main():
    out_dir = "/app/output"
    os.makedirs(out_dir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="frames_")
    try:
        for i in range(N_FRAMES):
            t = i / (N_FRAMES - 1)
            az = AZ0 + (AZ1 - AZ0) * t
            frame = render(az)
            if i == 0:
                # first frame must equal the given one exactly
                frame = Image.open("/app/first_frame.png").convert("RGB")
            frame.save(os.path.join(tmp, f"f{i:03d}.png"))
        out = os.path.join(out_dir, "video.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
            "-i", os.path.join(tmp, "f%03d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
            "-r", str(FPS), out,
        ], check=True)
        print("wrote", out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
