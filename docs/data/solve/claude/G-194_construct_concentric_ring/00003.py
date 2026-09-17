#!/usr/bin/env python3
"""Move two ring-shaped circles so they become concentric at the image center.

The rings are extracted as exact pixel masks from first_frame.png and
translated by integer offsets each frame, so their size, colour and stroke
are preserved exactly. Everything else (pure white background) is untouched.
"""
import os
import subprocess
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES = 40
FPS = 16


def main():
    img = np.array(Image.open(SRC).convert("RGB"))
    H, W, _ = img.shape
    target = np.array([W / 2.0, H / 2.0])  # (x, y) image center

    # Background colour = most common pixel (white here).
    cols, counts = np.unique(img.reshape(-1, 3), axis=0, return_counts=True)
    order = np.argsort(-counts)
    bg = cols[order[0]]
    shape_cols = [tuple(cols[i]) for i in order[1:3]]

    # Extract each ring as a mask + colour and its centroid (ring centre).
    shapes = []
    for col in shape_cols:
        m = np.all(img == np.array(col, dtype=np.uint8), axis=2)
        ys, xs = np.nonzero(m)
        centre = np.array([xs.mean(), ys.mean()])
        shapes.append(dict(color=np.array(col, dtype=np.uint8), mask=m, centre=centre))

    def ease(t):  # smooth ease-in-out
        return 0.5 - 0.5 * np.cos(np.pi * t)

    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for i in range(N_FRAMES):
        t = ease(i / (N_FRAMES - 1))
        frame = np.empty_like(img)
        frame[:] = bg
        # Draw larger ring first, smaller on top (they never overlap anyway).
        for s in sorted(shapes, key=lambda s: -s["mask"].sum()):
            off = np.rint((target - s["centre"]) * t).astype(int)  # (dx, dy)
            ys, xs = np.nonzero(s["mask"])
            xs2, ys2 = xs + off[0], ys + off[1]
            ok = (xs2 >= 0) & (xs2 < W) & (ys2 >= 0) & (ys2 < H)
            frame[ys2[ok], xs2[ok]] = s["color"]
        frames.append(frame)

    # Frame 0 must be identical to the source image.
    assert np.array_equal(frames[0], img), "first frame mismatch"

    raw = np.concatenate(frames).tobytes()
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "medium",
        "-movflags", "+faststart", OUT,
    ]
    subprocess.run(cmd, input=raw, check=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
