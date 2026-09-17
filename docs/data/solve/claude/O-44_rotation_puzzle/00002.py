#!/usr/bin/env python3
"""Solve the 2x2 rotating-pipe puzzle: rotate the elbow inside each tile so the
four pipe segments form one continuous (closed) path. Only the pipe pixels
inside the tile interiors change; every other pixel is copied from first_frame.png.
"""
import math
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

W = H = 1024
FPS = 16
N_FRAMES = 96
MOVE_END = 90          # rotation finishes here; remaining frames hold the solved state

PIPE = (59, 130, 246)
WHITE = (255, 255, 255)
HALF_W = 7             # pipe half width  -> 15 px wide
ARM_LEN = 103          # arm reaches 103 px from tile centre
ARM_BACK = 7           # arm rectangle starts 7 px behind centre (square joint)

# tile interiors (inclusive) and centres, measured from first_frame.png
TILES = {
    "TL": dict(x0=280, y0=280, x1=494, y1=494, cx=387, cy=387),
    "TR": dict(x0=530, y0=280, x1=744, y1=494, cx=637, cy=387),
    "BL": dict(x0=280, y0=530, x1=494, y1=744, cx=387, cy=637),
    "BR": dict(x0=530, y0=530, x1=744, y1=744, cx=637, cy=637),
}
# target arm directions (deg, image coords: 0=right, 90=down) for a closed loop
TARGET = {"TL": (0, 90), "TR": (180, 90), "BL": (270, 0), "BR": (270, 180)}


def arm_polygon(cx, cy, ang_deg):
    a = math.radians(ang_deg)
    dx, dy = math.cos(a), math.sin(a)
    px, py = -dy, dx
    pts = []
    for along, across in ((-ARM_BACK, -HALF_W), (ARM_LEN, -HALF_W),
                          (ARM_LEN, HALF_W), (-ARM_BACK, HALF_W)):
        pts.append((round(cx + dx * along + px * across), round(cy + dy * along + py * across)))
    return pts


def draw_pipe(img, tile, angles):
    """Erase the tile interior to white and draw the elbow with the given arm angles."""
    d = ImageDraw.Draw(img)
    d.rectangle([tile["x0"], tile["y0"], tile["x1"], tile["y1"]], fill=WHITE)
    for ang in angles:
        d.polygon(arm_polygon(tile["cx"], tile["cy"], ang), fill=PIPE)


def pipe_mask(first, tile):
    sub = first[tile["y0"]:tile["y1"] + 1, tile["x0"]:tile["x1"] + 1].astype(int)
    return (sub[:, :, 2] > 200) & (sub[:, :, 0] < 120)


def render_mask(tile, angles):
    img = Image.new("RGB", (W, H), WHITE)
    draw_pipe(img, tile, angles)
    sub = np.array(img)[tile["y0"]:tile["y1"] + 1, tile["x0"]:tile["x1"] + 1]
    return sub[:, :, 0] < 120


def fit_orientation(first, tile):
    """Find the elbow orientation (angle of first arm; second arm = +90) that best
    reproduces the pipe pixels in the first frame."""
    ref = pipe_mask(first, tile)
    best = None
    for step in (1.0, 0.1):
        rng = np.arange(0, 360, step) if best is None else np.arange(best - 1, best + 1, step)
        scores = []
        for ang in rng:
            m = render_mask(tile, (ang, ang + 90))
            scores.append(np.logical_xor(m, ref).sum())
        scores = np.array(scores)
        best = float(rng[int(np.argmin(scores))])
    best = round(best % 360, 1)
    # prefer an exactly axis-aligned orientation when it matches equally well
    snapped = round(best / 90) * 90 % 360
    snap_score = np.logical_xor(render_mask(tile, (snapped, snapped + 90)), ref).sum()
    if snap_score <= scores.min():
        best = float(snapped)
    return best


def shortest_rotation(start, target_pair):
    """Signed rotation that carries arms (start, start+90) onto target_pair, minimal magnitude."""
    cands = []
    for t in target_pair:
        rot = (t - start + 180) % 360 - 180
        got = {round((start + rot) % 360) % 360, round((start + 90 + rot) % 360) % 360}
        if got == {round(t) % 360 for t in target_pair}:
            cands.append(rot)
    return min(cands, key=abs)


def ease(t):
    # smooth ease-in-out (smootherstep)
    return t * t * t * (t * (6 * t - 15) + 10)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    first_img = Image.open(FIRST).convert("RGB")
    first = np.array(first_img)

    plan = {}
    for name, tile in TILES.items():
        start = fit_orientation(first, tile)
        rot = shortest_rotation(start, TARGET[name])
        plan[name] = (start, rot)
        print(f"{name}: start arm angle {start:.1f} deg, rotate {rot:+.1f} deg")

    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "8", "-preset", "slow", "-r", str(FPS), OUT],
        stdin=subprocess.PIPE)

    for f in range(N_FRAMES):
        if f == 0:
            frame = first_img.copy()
        else:
            t = min(1.0, f / MOVE_END)
            k = ease(t)
            frame = first_img.copy()
            for name, tile in TILES.items():
                start, rot = plan[name]
                if rot == 0:
                    continue  # already solved; leave pixels untouched
                ang = start + rot * k
                draw_pipe(frame, tile, (ang, ang + 90))
        ffmpeg.stdin.write(np.array(frame).tobytes())
        if f == N_FRAMES - 1:
            frame.save(os.path.join(OUT_DIR, "last_frame.png"))
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
