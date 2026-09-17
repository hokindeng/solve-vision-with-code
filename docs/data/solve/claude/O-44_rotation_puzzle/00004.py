#!/usr/bin/env python3
"""Generate the pipe-rotation puzzle solution video.

Scene (from first_frame.png): four 215x215 white tiles with 3px gray borders in a
2x2 grid; each tile holds a blue L-shaped elbow pipe (two 15px-wide arms from the
tile centre to the tile edge).  The pipes are rasterised exactly like the source
frame: two axis-aligned rectangles rotated about the tile centre, coordinates
rounded to integers, filled with PIL (no anti-aliasing).

Solution: rotate the elbows into a closed ring
    TL: right+down (already correct, stays)   TR: left+down
    BL: up+right                              BR: up+left
All pipes rotate simultaneously with an ease-in-out profile and finish together.
"""
import math, os, subprocess, shutil
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

W = H = 1024
FPS = 16
N_FRAMES = 96
MOVE_FRAMES = 88          # motion spans frames 0..MOVE_FRAMES-1, then hold

BLUE = (14, 165, 233)
WHITE = (255, 255, 255)
TILE = 215                # inner (white) tile size
CX = CY = 107             # rotation centre in tile-local coords

# tile-local pixel origin (top-left of white interior), start angle, end angle.
# Angle convention: degrees, positive = clockwise on screen; 0 = arms right+down.
TILES = {
    "TL": ((280, 280), 0.0, 0.0),
    "TR": ((530, 280), 190.25, 90.0),
    "BL": ((280, 530), 80.15, 270.0),
    "BR": ((530, 530), 5.25, 180.0),
}
ARMS = [
    [(100, 100), (210, 100), (210, 114), (100, 114)],   # arm pointing right
    [(100, 100), (114, 100), (114, 210), (100, 210)],   # arm pointing down
]


def shortest_delta(a, b):
    d = (b - a) % 360.0
    if d > 180.0:
        d -= 360.0
    return d


def ease(t):
    """smoothstep ease-in-out"""
    return t * t * (3.0 - 2.0 * t)


def draw_tile(angle):
    """Return a 215x215 RGB tile with the elbow pipe at the given angle."""
    im = Image.new("RGB", (TILE, TILE), WHITE)
    d = ImageDraw.Draw(im)
    t = math.radians(angle)
    c, s = math.cos(t), math.sin(t)
    for pts in ARMS:
        poly = [(round(CX + (x - CX) * c - (y - CY) * s),
                 round(CY + (x - CX) * s + (y - CY) * c)) for x, y in pts]
        d.polygon(poly, fill=BLUE)
    return np.array(im)


def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    assert base.shape == (H, W, 3)

    tmp = os.path.join(OUT_DIR, "_frames")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)

    for i in range(N_FRAMES):
        if i == 0:
            frame = base.copy()                      # exact first frame
        else:
            p = min(1.0, i / (MOVE_FRAMES - 1))
            e = ease(p)
            frame = base.copy()
            for name, ((x0, y0), a0, a1) in TILES.items():
                delta = shortest_delta(a0, a1)
                if delta == 0.0:
                    continue                          # TL already solved
                ang = a0 + delta * e
                frame[y0:y0 + TILE, x0:x0 + TILE] = draw_tile(ang)
        Image.fromarray(frame).save(os.path.join(tmp, f"{i:04d}.png"))

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp, "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "10", "-preset", "slow",
        "-movflags", "+faststart", OUT,
    ], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
