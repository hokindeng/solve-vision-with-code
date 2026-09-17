#!/usr/bin/env python3
"""Draw a red circle around the single Chinese character (习) in first_frame.png,
animated as a stroke that sweeps around the character over 3 seconds."""
import math
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

FPS = 16
N_FRAMES = 48
W = H = 1024

# Bounding box of the Chinese character 习 (measured from first_frame.png).
X0, X1, Y0, Y1 = 561, 669, 177, 294
CX, CY = (X0 + X1) / 2.0, (Y0 + Y1) / 2.0
RADIUS = math.hypot(X1 - X0, Y1 - Y0) / 2 + 14  # comfortably encloses the glyph
LINE_W = 7
RED = (220, 20, 20)
SCALE = 4  # supersample for a smooth anti-aliased stroke


def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * t)


def draw_arc_overlay(frac):
    """Return an RGBA overlay with the circle drawn up to `frac` of the way round."""
    big = Image.new("RGBA", (W * SCALE, H * SCALE), (0, 0, 0, 0))
    if frac <= 0:
        return big.resize((W, H), Image.LANCZOS)
    d = ImageDraw.Draw(big)
    r = RADIUS * SCALE
    box = [CX * SCALE - r, CY * SCALE - r, CX * SCALE + r, CY * SCALE + r]
    start = -90.0  # begin at the top, sweep clockwise
    end = start + 360.0 * min(frac, 1.0)
    lw = LINE_W * SCALE
    if frac >= 1.0:
        d.ellipse(box, outline=RED + (255,), width=lw)
    else:
        d.arc(box, start=start, end=end, fill=RED + (255,), width=lw)
        # round caps on the stroke ends
        for ang in (start, end):
            a = math.radians(ang)
            px, py = CX * SCALE + r * math.cos(a), CY * SCALE + r * math.sin(a)
            hw = lw / 2
            d.ellipse([px - hw, py - hw, px + hw, py + hw], fill=RED + (255,))
    return big.resize((W, H), Image.LANCZOS)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = Image.open(FIRST).convert("RGB")
    assert base.size == (W, H)

    frames_dir = os.path.join(OUT_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Timeline: frame 0 untouched; the circle is drawn from frame 1 through frame 40
    # and then holds complete for the remaining frames.
    draw_start, draw_end = 1, 40
    for i in range(N_FRAMES):
        if i < draw_start:
            frac = 0.0
        elif i >= draw_end:
            frac = 1.0
        else:
            frac = ease((i - draw_start + 1) / (draw_end - draw_start + 1))
        frame = base.copy()
        if frac > 0:
            overlay = draw_arc_overlay(frac)
            frame = Image.alpha_composite(frame.convert("RGBA"), overlay).convert("RGB")
        frame.save(os.path.join(frames_dir, f"{i:04d}.png"))

    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(FPS),
            "-i", os.path.join(frames_dir, "%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
            "-r", str(FPS),
            OUT,
        ],
        check=True,
    )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
