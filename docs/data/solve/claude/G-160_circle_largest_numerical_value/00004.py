#!/usr/bin/env python3
"""Draw a red circle, step by step, around the largest number (88) in first_frame.png."""
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
N_FRAMES = 80
SIZE = 1024
SS = 4  # supersampling factor for a smooth stroke

# Numbers in the scene, located from the pixel data: value -> bbox (x0, y0, x1, y1)
NUMBERS = {
    59: (259, 269, 408, 359),
    85: (679, 258, 829, 348),
    62: (463, 507, 611, 597),
    88: (619, 774, 770, 864),
}

# Step 1: compare the numerical values and pick the single largest one.
largest = max(NUMBERS)
x0, y0, x1, y1 = NUMBERS[largest]
CX, CY = (x0 + x1) / 2.0, (y0 + y1) / 2.0
RADIUS = 105.0
STROKE = 7
COLOR = (220, 20, 20, 255)

# Timing: short pause while "comparing", then the circle is drawn progressively,
# then hold the finished result.
START = 14          # first frame where the pen starts drawing
END = 68            # frame at which the circle is complete
START_ANGLE = -90.0  # begin at the top, draw clockwise


def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * t)


def arc_layer(sweep_deg):
    """RGBA layer (SIZE x SIZE) containing the partial arc, transparent elsewhere."""
    big = Image.new("RGBA", (SIZE * SS, SIZE * SS), (0, 0, 0, 0))
    if sweep_deg <= 0:
        return big.resize((SIZE, SIZE), Image.LANCZOS)
    d = ImageDraw.Draw(big)
    r, w = RADIUS * SS, STROKE * SS
    cx, cy = CX * SS, CY * SS
    bbox = [cx - r, cy - r, cx + r, cy + r]
    if sweep_deg >= 360.0:
        # completed circle: draw a seamless ring
        d.ellipse(bbox, outline=COLOR, width=w)
        return big.resize((SIZE, SIZE), Image.LANCZOS)
    end = START_ANGLE + sweep_deg
    d.arc(bbox, START_ANGLE, end, fill=COLOR, width=w)
    # round pen caps at both ends of the stroke
    for ang in (START_ANGLE, end):
        a = math.radians(ang)
        px, py = cx + r * math.cos(a), cy + r * math.sin(a)
        hw = w / 2.0
        d.ellipse([px - hw, py - hw, px + hw, py + hw], fill=COLOR)
    return big.resize((SIZE, SIZE), Image.LANCZOS)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = Image.open(FIRST).convert("RGBA")
    assert base.size == (SIZE, SIZE)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SIZE}x{SIZE}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "8", "-preset", "slow", "-tune", "stillimage",
        "-r", str(FPS), OUT,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(N_FRAMES):
        if i < START:
            frame = base
        else:
            t = min(1.0, (i - START) / float(END - START))
            sweep = 360.0 * ease(t)
            frame = Image.alpha_composite(base, arc_layer(sweep))
        proc.stdin.write(np.asarray(frame.convert("RGB"), dtype=np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {OUT}: circled {largest} at ({CX:.1f}, {CY:.1f}), r={RADIUS}")


if __name__ == "__main__":
    main()
