#!/usr/bin/env python3
"""Move two ring-shaped circles so they become concentric at the image center."""
import os
import subprocess
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

W = H = 1024
FPS = 16
N_FRAMES = 40
BG = (255, 255, 255)
LINE_W = 8

# Measured from first_frame.png (Pillow ellipse, outline width 8 reproduces it exactly).
CIRCLES = [
    # color, start center (x, y), radius
    ((128, 128, 0), (437, 307), 423),   # large olive ring (top part clipped initially)
    ((245, 130, 48), (910, 716), 100),  # small orange ring
]
TARGET = (W // 2, H // 2)


def ease(t):
    """Smoothstep ease-in-out."""
    return t * t * (3 - 2 * t)


def draw_frame(t):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    s = ease(t)
    for color, (x0, y0), r in CIRCLES:
        cx = int(round(x0 + (TARGET[0] - x0) * s))
        cy = int(round(y0 + (TARGET[1] - y0) * s))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=LINE_W)
    return img


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = [draw_frame(i / (N_FRAMES - 1)) for i in range(N_FRAMES)]

    # Sanity: first frame must match the reference exactly.
    ref = np.array(Image.open(FIRST).convert("RGB"))
    diff = int((np.array(frames[0]) != ref).any(axis=2).sum())
    if diff:
        print(f"warning: first frame differs from reference in {diff} pixels")
        frames[0] = Image.fromarray(ref)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-movflags", "+faststart", OUT,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.array(f, dtype=np.uint8).tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {OUT} ({N_FRAMES} frames @ {FPS} fps)")


if __name__ == "__main__":
    main()
