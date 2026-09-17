#!/usr/bin/env python3
"""Domino chain reaction animation generated from first_frame.png.

Every pixel outside the falling dominos is copied verbatim from the first
frame. Each domino topples to the right (perspective flatten: the top edge
slides right and down onto the base), pivoting on its bottom edge.
"""
import os
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT = os.path.join(HERE, "output", "video.mp4")

W = H = 1024
FPS = 16
N_FRAMES = 62
BG = (240, 235, 230)

# Standing domino boxes (x0, y0, x1, y1) inclusive, measured on first_frame.png.
# START's label overflows its box, so its crop is 2 px wider on each side.
DOMINOS = {
    "START": (42, 438, 110, 531),
    "T1": (169, 438, 232, 531),
    "T2": (294, 438, 357, 531),
    "T3": (419, 438, 482, 531),
    "A1": (544, 341, 607, 434),
    "A2": (669, 316, 732, 409),
    "B1": (544, 535, 607, 628),
    "B2": (669, 560, 732, 653),
    "B3": (794, 584, 857, 677),
    "B4": (919, 609, 982, 702),
}
# A3, A4 never fall (gap in Branch A).

FALL_LEN = 11          # frames a single domino takes to fall
STAGGER = 6            # frames between consecutive triggers
FIRST_TRIGGER = 2      # frame at which START gets pushed
ORDER = [["START"], ["T1"], ["T2"], ["T3"], ["A1", "B1"], ["A2", "B2"], ["B3"], ["B4"]]
START_FRAME = {}
for i, group in enumerate(ORDER):
    for name in group:
        START_FRAME[name] = FIRST_TRIGGER + i * STAGGER

THICK = 14             # apparent thickness of a domino lying flat
LEAN = 0.32            # how far (fraction of height) the top edge slides right


def ease(t):
    """Gravity-like acceleration with a small settle at the end."""
    t = min(max(t, 0.0), 1.0)
    return t ** 1.9


def domino_progress(name, frame):
    s = START_FRAME[name]
    if frame <= s:
        return 0.0
    return ease((frame - s) / FALL_LEN)


def main():
    base = Image.open(FIRST).convert("RGB")
    base_np = np.array(base)
    assert base.size == (W, H)

    crops = {}
    for name, (x0, y0, x1, y1) in DOMINOS.items():
        crop = base.crop((x0, y0, x1 + 1, y1 + 1)).convert("RGBA")
        crops[name] = crop

    tmp = tempfile.mkdtemp(prefix="domino_frames_")
    try:
        for f in range(N_FRAMES):
            frame_np = base_np.copy()
            active = []
            for name, (x0, y0, x1, y1) in DOMINOS.items():
                p = domino_progress(name, f)
                if p > 0.0:
                    # erase the standing domino; it will be redrawn tilted
                    frame_np[y0:y1 + 1, x0:x1 + 1] = BG
                    active.append((name, p))
            frame = Image.fromarray(frame_np).convert("RGBA")

            # draw the rear-most (right-most) ones first so leaning tops of
            # earlier dominos overlap correctly
            for name, p in sorted(active, key=lambda a: -DOMINOS[a[0]][0]):
                x0, y0, x1, y1 = DOMINOS[name]
                crop = crops[name]
                cw, ch = crop.size
                theta = p * np.pi / 2
                h = ch * np.cos(theta) + THICK * np.sin(theta)
                dx = ch * LEAN * np.sin(theta)
                # output (x, y) -> source (u, v)
                yb = y1 + 1            # bottom edge (exclusive)
                top = yb - h
                e = ch / h
                fcoef = -e * top
                a = 1.0
                b = dx / h
                c = -x0 - dx - (dx / h) * top
                warped = crop.transform((W, H), Image.AFFINE, (a, b, c, 0.0, e, fcoef),
                                        resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))
                frame = Image.alpha_composite(frame, warped)

            frame = frame.convert("RGB")
            if f == 0:
                # first frame must be exactly the given one
                frame = base.copy()
            frame.save(os.path.join(tmp, f"{f:04d}.png"))

        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(FPS), "-i", os.path.join(tmp, "%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
            "-preset", "slow", "-r", str(FPS), OUT,
        ]
        subprocess.run(cmd, check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
