"""Highlight the maximum point of the area chart with a red rectangular border.

The border is traced progressively (clockwise, starting top-left) over 48 frames,
so frame 0 equals first_frame.png and the final frame shows the complete box.
"""
import os
import shutil
import subprocess
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

W = H = 1024
FPS = 16
N_FRAMES = 48

# Max point (value 62): teal marker centred at ~(552, 226), radius ~15 px.
CX, CY, R = 552, 226, 15
# The "62" label (x 535-568, y 184-211) touches the marker top, so the box
# encloses marker + label rather than slicing through the text.
PAD = 8
X0, X1 = CX - R - PAD, CX + R + PAD          # 529 .. 575
Y0, Y1 = 184 - 7, CY + R + PAD               # 177 .. 249
RED = (220, 20, 20)
THICK = 3

# Clockwise polyline of the border, starting at the top-left corner.
CORNERS = [(X0, Y0), (X1, Y0), (X1, Y1), (X0, Y1), (X0, Y0)]
SEG_LEN = [abs(b[0] - a[0]) + abs(b[1] - a[1]) for a, b in zip(CORNERS, CORNERS[1:])]
TOTAL = sum(SEG_LEN)


def partial_border(base: Image.Image, t: float) -> Image.Image:
    """Draw the fraction t (0..1) of the border perimeter onto a copy of base."""
    img = base.copy()
    if t <= 0:
        return img
    dr = ImageDraw.Draw(img)
    remaining = t * TOTAL
    for (ax, ay), (bx, by), L in zip(CORNERS, CORNERS[1:], SEG_LEN):
        if remaining <= 0:
            break
        f = min(1.0, remaining / L)
        ex = ax + (bx - ax) * f
        ey = ay + (by - ay) * f
        dr.line([(ax, ay), (ex, ey)], fill=RED, width=THICK)
        remaining -= L
    if t >= 1.0:
        # Clean, fully-joined rectangle on the final state.
        dr.rectangle([X0, Y0, X1, Y1], outline=RED, width=THICK)
    return img


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    base = Image.open(SRC).convert("RGB")
    assert base.size == (W, H)

    frames_dir = os.path.join(OUT_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for i in range(N_FRAMES):
        # Ease so the tracing spans the full duration; frame 0 -> 0, last -> 1.
        t = i / (N_FRAMES - 1)
        t = t * t * (3 - 2 * t)  # smoothstep
        frame = partial_border(base, t)
        frame.save(os.path.join(frames_dir, f"{i:04d}.png"))

    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(FPS),
            "-i", os.path.join(frames_dir, "%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
            "-r", str(FPS),
            OUT,
        ],
        check=True,
    )
    shutil.rmtree(frames_dir, ignore_errors=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
