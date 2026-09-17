#!/usr/bin/env python3
"""Animate a red rectangular border around the minimum-value bar of the chart.

Frame 0 is the untouched first_frame.png. Over the next frames a red border is
traced clockwise around the minimum bar (the "35" bar), then held to the end.
Every pixel outside the border stays identical to the first frame.
"""
import subprocess
import numpy as np
from PIL import Image

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
W = H = 1024
FPS = 16
N_FRAMES = 48
DRAW_FRAMES = 42          # frames spent tracing the border; the rest hold

# Pixel bounds of the minimum bar (value 35) including its black outline,
# measured from first_frame.png: x 221..289, y 772..873.
BAR_X0, BAR_X1, BAR_Y0, BAR_Y1 = 221, 289, 772, 873
PAD = 2                   # gap between the bar outline and the red border
THICK = 3                 # border thickness in pixels
RED = np.array([230, 20, 20], dtype=np.uint8)

# Outer rectangle of the border (inclusive coordinates)
X0, X1 = BAR_X0 - PAD - THICK, BAR_X1 + PAD + THICK
Y0, Y1 = BAR_Y0 - PAD - THICK, BAR_Y1 + PAD + THICK


def border_mask(progress):
    """Boolean mask of the border drawn up to `progress` in [0, 1].

    The border is traced clockwise starting at the top-left corner:
    top edge -> right edge -> bottom edge -> left edge.
    """
    mask = np.zeros((H, W), dtype=bool)
    if progress <= 0:
        return mask
    w = X1 - X0 + 1
    h = Y1 - Y0 + 1
    perim = 2 * w + 2 * h
    length = progress * perim

    # top edge, left -> right
    seg = min(length, w)
    if seg > 0:
        mask[Y0:Y0 + THICK, X0:X0 + int(round(seg))] = True
    length -= w
    # right edge, top -> bottom
    if length > 0:
        seg = min(length, h)
        mask[Y0:Y0 + int(round(seg)), X1 - THICK + 1:X1 + 1] = True
        length -= h
    # bottom edge, right -> left
    if length > 0:
        seg = min(length, w)
        mask[Y1 - THICK + 1:Y1 + 1, X1 + 1 - int(round(seg)):X1 + 1] = True
        length -= w
    # left edge, bottom -> top
    if length > 0:
        seg = min(length, h)
        mask[Y1 + 1 - int(round(seg)):Y1 + 1, X0:X0 + THICK] = True
    return mask


def ease(t):
    # smoothstep for a gentle start/finish
    return t * t * (3 - 2 * t)


def main():
    base = np.array(Image.open(SRC).convert("RGB"))
    assert base.shape == (H, W, 3)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
        "-preset", "slow", "-movflags", "+faststart",
        OUT,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(N_FRAMES):
        frame = base.copy()
        if i > 0:
            t = min(1.0, i / (DRAW_FRAMES - 1))
            mask = border_mask(ease(t))
            frame[mask] = RED
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
