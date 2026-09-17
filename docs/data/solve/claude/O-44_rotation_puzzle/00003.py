#!/usr/bin/env python3
"""Rotate the pipe inside each tile so that all four elbows join into a closed loop.

Geometry measured from first_frame.png:
  * tiles: 221x221 squares at (277,277), (527,277), (277,527), (527,527); 3px border,
    white interior 280..494 (x and y, relative to the first tile).
  * each pipe is an L-shaped elbow: two 15px-thick arms of length 103 from the tile
    centre, drawn in green (132,204,22). Orientation theta => arms at theta and theta+90.
"""
import math, os, subprocess
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

FPS = 16
N_FRAMES = 96
MOVE_FRAMES = 80          # rotation happens over frames 0..80, then hold (puzzle solved)

GREEN = (132, 204, 22)
WHITE = (255, 255, 255)
ARM_LEN = 103
HALF_W = 7
SS = 4                    # supersampling factor for smooth edges

# tile top-left corner, pipe pivot (tile centre), start angle, target angle (deg)
TILES = [
    dict(origin=(277, 277), start=0.0,   target=0.0),    # TL: right + down (already solved)
    dict(origin=(527, 277), start=-78.0, target=90.0),   # TR: needs down + left
    dict(origin=(277, 527), start=-9.0,  target=-90.0),  # BL: needs up + right
    dict(origin=(527, 527), start=27.0,  target=180.0),  # BR: needs left + up
]


def shortest_delta(a, b):
    d = (b - a) % 360.0
    if d > 180.0:
        d -= 360.0
    return d


def ease(t):
    # smooth ease-in-out
    return 0.5 - 0.5 * math.cos(math.pi * t)


def pipe_layer(theta_deg, size=221):
    """Return an (size,size) uint8 alpha mask of the elbow at angle theta, pivot at centre."""
    # pixel-centre convention: the source pipe covers pixels 380..394 (15px) and 380..490,
    # i.e. continuous extents [-7.5, 7.5) and [-7.5, 103.5) around centre 387.5.
    c = 110.5
    eps = 0.5 / SS                     # PIL fills polygon edges inclusively; pull in half a subpixel
    hw, al = HALF_W + 0.5 - eps, ARM_LEN + 0.5 - eps
    pts = [(-hw, -hw), (al, -hw), (al, hw), (hw, hw), (hw, al), (-hw, al)]
    t = math.radians(theta_deg)
    ct, st = math.cos(t), math.sin(t)
    P = [((c + x * ct - y * st) * SS, (c + x * st + y * ct) * SS) for x, y in pts]
    im = Image.new("L", (size * SS, size * SS), 0)
    ImageDraw.Draw(im).polygon(P, fill=255)
    im = im.resize((size, size), Image.BOX)
    return np.asarray(im).astype(np.float32) / 255.0


def render_frame(base, thetas):
    frame = base.copy()
    for tile, th in zip(TILES, thetas):
        ox, oy = tile["origin"]
        alpha = pipe_layer(th)
        # only touch the white interior (3px border stays untouched)
        sl = (slice(oy + 3, oy + 218), slice(ox + 3, ox + 218))
        a = alpha[3:218, 3:218][..., None]
        region = np.full((215, 215, 3), WHITE, dtype=np.float32)
        region = region * (1 - a) + np.array(GREEN, dtype=np.float32) * a
        frame[sl] = np.clip(region + 0.5, 0, 255).astype(np.uint8)
    return frame


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = np.asarray(Image.open(FIRST).convert("RGB")).copy()
    deltas = [shortest_delta(t["start"], t["target"]) for t in TILES]

    frames = []
    for i in range(N_FRAMES):
        if i == 0:
            frames.append(base)          # first frame is exactly first_frame.png
            continue
        s = ease(min(1.0, i / MOVE_FRAMES))
        thetas = [t["start"] + d * s for t, d in zip(TILES, deltas)]
        frames.append(render_frame(base, thetas))

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "1024x1024", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "medium",
           "-r", str(FPS), OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print("wrote", OUT, len(frames), "frames")


if __name__ == "__main__":
    main()
