#!/usr/bin/env python3
"""Move the green attention box from the left object to the right object."""
import os, subprocess, tempfile
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT = os.path.join(HERE, "output", "video.mp4")
N_FRAMES, FPS = 25, 16
BOX_COLOR = np.array([70, 140, 70], dtype=np.uint8)


def main():
    first = np.array(Image.open(FIRST).convert("RGB"))
    box_mask = np.all(first == BOX_COLOR, axis=-1)
    ys, xs = np.where(box_mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    box_cx = (x0 + x1) / 2.0

    # Right object: the circle (fill color 94,156,86 with a dark outline).
    obj = np.all(first == np.array([94, 156, 86], dtype=np.uint8), axis=-1)
    oys, oxs = np.where(obj)
    obj_cx = (oxs.min() + oxs.max()) / 2.0
    dx_total = int(round(obj_cx - box_cx))  # 569

    # Background: first frame with the box erased (box sits on pure white).
    bg = first.copy()
    bg[box_mask] = 255
    box_patch = box_mask[y0:y1 + 1, x0:x1 + 1]

    frames = []
    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)
        s = 0.5 - 0.5 * np.cos(np.pi * t)  # ease in/out
        dx = int(round(dx_total * s))
        fr = bg.copy()
        region = fr[y0:y1 + 1, x0 + dx:x1 + 1 + dx]
        region[box_patch] = BOX_COLOR
        frames.append(fr)

    # Frame 0 must be identical to first_frame.png.
    assert np.array_equal(frames[0], first)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        for i, fr in enumerate(frames):
            Image.fromarray(fr).save(os.path.join(td, f"f{i:03d}.png"))
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
            "-i", os.path.join(td, "f%03d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
            "-preset", "slow", "-r", str(FPS), OUT,
        ], check=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
