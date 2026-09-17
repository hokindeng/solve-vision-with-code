#!/usr/bin/env python3
"""Animate the answer to the shape-scaling analogy A:B :: C:?.

A (L-shape, 151 px box) -> B (L-shape, 131 px box): factor = 131/151 ~ 0.867 (130/150).
C (trapezoid, 151 px box) -> ? : same factor, centred where the '?' sits.
"""
import subprocess
import numpy as np
from PIL import Image, ImageDraw

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS, N = 16, 60

base = np.array(Image.open(SRC).convert("RGB"))
H, W = base.shape[:2]

# --- measurements from first_frame.png ---------------------------------
# A outline bbox: x 180..330, y 181..331  (151 px)
# B outline bbox: x 704..834, y 191..321  (131 px)
FACTOR = 131.0 / 151.0
# C trapezoid outline vertices (x, y)
C_VERTS = np.array([(218, 693), (292, 693), (330, 843), (180, 843)], float)
C_CENTER = np.array([255.0, 768.0])
# '?' glyph bbox: x 756..782, y 746..789 ; its slot is centred like B: (769, 768)
Q_BBOX = (756, 782, 746, 789)
Q_CENTER = np.array([769.0, 768.0])

BLUE, BLACK = (0, 0, 255), (0, 0, 0)


def smooth(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def trapezoid(scale):
    """Vertices of C scaled by `scale` about its centre, moved to the ? slot."""
    return (C_VERTS - C_CENTER) * scale + Q_CENTER


def draw_shape(img, verts):
    d = ImageDraw.Draw(img)
    d.polygon([tuple(v) for v in verts], fill=BLUE, outline=BLACK)


def make_frame(i):
    frame = base.copy()
    # Phase 1 (frames 0..12): '?' fades to white.
    q_alpha = 1.0 - smooth(i / 12.0)
    x0, x1, y0, y1 = Q_BBOX
    patch = frame[y0:y1 + 1, x0:x1 + 1].astype(float)
    frame[y0:y1 + 1, x0:x1 + 1] = np.round(255 + (patch - 255) * q_alpha).astype(np.uint8)

    # Phase 2 (frames 12..50): a copy of C appears in the slot and shrinks to the
    # inferred factor; then hold to the end.
    if i >= 12:
        t = smooth((i - 12) / 38.0)
        scale = 1.0 + (FACTOR - 1.0) * t
        fade = smooth((i - 12) / 6.0)
        shape_img = Image.fromarray(frame.copy())
        draw_shape(shape_img, trapezoid(scale))
        shape = np.array(shape_img).astype(float)
        frame = np.round(frame + (shape - frame) * fade).astype(np.uint8)
    return frame


def main():
    frames = [make_frame(i) for i in range(N)]
    # first frame must be identical to the source image
    assert np.array_equal(frames[0], base)
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "slow", OUT],
        stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()
    Image.fromarray(frames[-1]).save("/app/output/last_frame.png")


if __name__ == "__main__":
    main()
