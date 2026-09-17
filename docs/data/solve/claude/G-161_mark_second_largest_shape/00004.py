#!/usr/bin/env python3
"""Find the second-largest circle in first_frame.png and animate a red ring around it."""
import os
import subprocess
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 40


def find_circles(img):
    """Return list of (area, cx, cy, radius) for non-white blobs."""
    a = np.asarray(img.convert("RGB")).astype(int)
    mask = (np.abs(a - 255).sum(axis=2) > 30)  # anything not background
    mask = ndimage.binary_fill_holes(mask)
    labels, n = ndimage.label(mask)
    shapes = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(labels == i)
        area = len(xs)
        if area < 200:
            continue
        cx, cy = xs.mean(), ys.mean()
        r = (xs.max() - xs.min() + ys.max() - ys.min()) / 4.0
        shapes.append((area, cx, cy, r))
    return shapes


def main():
    base = Image.open(FIRST).convert("RGB")
    shapes = sorted(find_circles(base), key=lambda s: -s[0])
    assert len(shapes) >= 2, shapes
    area, cx, cy, r = shapes[1]  # second largest
    ring_r = r + 22
    width = 8
    bbox = [cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r]

    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for k in range(N_FRAMES):
        f = base.copy()
        # Frame 0 identical to first_frame; ring sweeps in, finishing a few frames before the end.
        t = k / (N_FRAMES - 4)
        t = min(max(t, 0.0), 1.0)
        if t > 0:
            ease = 1 - (1 - t) ** 2  # ease-out
            sweep = 360.0 * ease
            d = ImageDraw.Draw(f)
            start = -90
            if sweep >= 359.9:
                d.ellipse(bbox, outline=(255, 0, 0), width=width)
            else:
                d.arc(bbox, start, start + sweep, fill=(255, 0, 0), width=width)
        frames.append(f)

    # Encode with ffmpeg via raw RGB pipe.
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1024x1024", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.asarray(f, dtype=np.uint8).tobytes())
    p.stdin.close()
    p.wait()
    assert p.returncode == 0, "ffmpeg failed"
    print(f"second largest: center=({cx:.1f},{cy:.1f}) r={r:.1f} area={area}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
