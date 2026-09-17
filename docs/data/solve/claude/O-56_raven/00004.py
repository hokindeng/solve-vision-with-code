#!/usr/bin/env python3
"""Complete the 3x3 progressive matrix in first_frame.png.

Rule: each row/column is a permutation of {star, chevron, cross}
(Latin square). Row 3 has cross, star -> missing cell is the black chevron.
The video fades out the "?" then strokes in the chevron with an angular sweep.
"""
import os
import subprocess
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS = 16
N_FRAMES = 35

base = np.array(Image.open(SRC).convert("RGB"))
H, W, _ = base.shape

# Source chevron (top-middle cell) bounding box, measured from the image.
SY0, SY1, SX0, SX1 = 140, 228, 453, 569
DX, DY = 341, 682  # cell pitch: one column right, two rows down
TY0, TY1, TX0, TX1 = SY0 + DY, SY1 + DY, SX0 + DX, SX1 + DX

patch = base[SY0:SY1, SX0:SX1].astype(np.float32)
shape_mask = (np.abs(patch - 255).sum(2) > 30)

# The "?" glyph region in the bottom-right cell.
QY0, QY1, QX0, QX1 = 811, 893, 826, 878
q_region = base[QY0:QY1, QX0:QX1].astype(np.float32)
white = np.full_like(q_region, 255.0)

# Angular ordering of the shape pixels around its centre, for a stroke-like reveal.
ph, pw = patch.shape[:2]
yy, xx = np.mgrid[0:ph, 0:pw]
cy, cx = (ph - 1) / 2.0, (pw - 1) / 2.0
ang = (np.arctan2(yy - cy, xx - cx) + np.pi / 2) % (2 * np.pi)  # start at top, go clockwise
ang = ang / (2 * np.pi)

FADE_END = 9          # frames 1..FADE_END fade out the "?"
DRAW_START = 8        # drawing begins slightly overlapping the fade
DRAW_END = N_FRAMES - 1


def frame(i):
    f = base.astype(np.float32).copy()
    # Fade the question mark towards white.
    t = np.clip(i / FADE_END, 0.0, 1.0)
    f[QY0:QY1, QX0:QX1] = q_region * (1 - t) + white * t
    # Reveal the chevron by angular sweep.
    if i >= DRAW_START:
        s = np.clip((i - DRAW_START) / (DRAW_END - DRAW_START), 0.0, 1.0)
        reveal = shape_mask & (ang <= s + 1e-6)
        tgt = f[TY0:TY1, TX0:TX1]
        tgt[reveal] = patch[reveal]
    return np.clip(f + 0.5, 0, 255).astype(np.uint8)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = [frame(i) for i in range(N_FRAMES)]
    assert np.array_equal(frames[0], base), "first frame must match first_frame.png"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "slow", "-crf", "12",
        "-r", str(FPS), OUT,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        raise SystemExit("ffmpeg failed")
    Image.fromarray(frames[-1]).save(os.path.join(OUT_DIR, "last_frame.png"))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
