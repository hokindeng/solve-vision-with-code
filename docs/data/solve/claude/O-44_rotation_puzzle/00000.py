#!/usr/bin/env python3
"""Rotate the pipe segment inside each of the four tiles so they form one
continuous (closed-loop) path. Everything except the pipes is left untouched."""
import math, os, subprocess
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

W = H = 1024
FPS = 16
N_FRAMES = 96
MOVE_FRAMES = 90          # rotation happens over frames 0..MOVE_FRAMES, then hold

TEAL = (20, 184, 166)
WHITE = (255, 255, 255)
HALF_W, ARM_LEN = 7, 103  # pipe half-width and arm length (px) from the tile centre
INNER = 107               # half-size of the white tile interior (280..494 around centre 387)

# tile centres (pixel index of the centre pixel) and solved orientations.
# Local pipe shape: arms pointing +x (right) and +y (down); angle rotates clockwise on screen.
TILES = {
    "TL": ((387, 387), 0.0),    # right + down
    "TR": ((637, 387), 90.0),   # down + left
    "BR": ((637, 637), 180.0),  # left + up
    "BL": ((387, 637), 270.0),  # up + right
}


def pipe_mask(cx, cy, deg):
    """Boolean mask of the L-shaped pipe rotated by `deg` about (cx, cy)."""
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    h, L = HALF_W, ARM_LEN
    poly = [(-h, -h), (L, -h), (L, h), (h, h), (h, L), (-h, L)]
    pts = [(cx + x * c - y * s, cy + x * s + y * c) for x, y in poly]
    im = Image.new("L", (W, H), 0)
    ImageDraw.Draw(im).polygon(pts, fill=255)
    return np.array(im) > 0


def fit_angle(teal, cx, cy):
    """Find the current rotation of a tile's pipe by matching the first frame."""
    y0, y1, x0, x1 = cy - INNER, cy + INNER + 1, cx - INNER, cx + INNER + 1
    sub = teal[y0:y1, x0:x1]

    def cost(d):
        return (pipe_mask(cx, cy, d)[y0:y1, x0:x1] != sub).sum()

    best = min(range(0, 360), key=cost)
    fine = min((best + k / 10.0 for k in range(-10, 11)), key=cost)
    return fine % 360.0


def smoothstep(t):
    return t * t * (3 - 2 * t)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    src = np.array(Image.open(SRC).convert("RGB"))
    teal = np.all(src == TEAL, axis=2)

    start, delta = {}, {}
    for name, ((cx, cy), target) in TILES.items():
        a0 = fit_angle(teal, cx, cy)
        d = (target - a0) % 360.0        # rotate clockwise to the solved orientation
        if d < 1e-6:
            d = 0.0                       # already solved: stays put
        start[name], delta[name] = a0, d
        print(f"{name}: start {a0:.1f} deg -> target {target:.1f} deg (rotate {d:.1f})")

    # background with the pipe interiors cleared to white
    base = src.copy()
    for (cx, cy), _ in TILES.values():
        base[cy - INNER:cy + INNER + 1, cx - INNER:cx + INNER + 1] = WHITE

    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-r", str(FPS), OUT],
        stdin=subprocess.PIPE)

    last = None
    for i in range(N_FRAMES):
        if i == 0:
            frame = src.copy()            # first frame is exactly the given image
        else:
            t = smoothstep(min(1.0, i / MOVE_FRAMES))
            frame = base.copy()
            for name, ((cx, cy), _) in TILES.items():
                ang = start[name] + delta[name] * t
                frame[pipe_mask(cx, cy, ang)] = TEAL
        ffmpeg.stdin.write(frame.tobytes())
        last = frame
    ffmpeg.stdin.close()
    ffmpeg.wait()
    Image.fromarray(last).save(os.path.join(OUT_DIR, "last_frame.png"))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
