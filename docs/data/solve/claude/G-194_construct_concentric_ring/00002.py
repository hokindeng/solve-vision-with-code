#!/usr/bin/env python3
"""Regenerate /app/output/video.mp4: two ring circles translate to become
concentric at the image centre. Sizes, colours and background are unchanged."""
import os
import subprocess
import numpy as np
from PIL import Image, ImageDraw

W = H = 1024
FPS = 16
N_FRAMES = 40
COLOR = (184, 134, 11)
STROKE = 8
# Parameters recovered from first_frame.png (exact pixel match).
CIRCLES = [  # (cx, cy, r)
    (462, 307, 448),
    (787, 716, 223),
]
TARGET = (W // 2, H // 2)  # (512, 512) - image centre

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output")
FRAMES_DIR = os.path.join(OUT_DIR, "frames")


def render(centers):
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    for (cx, cy), (_, _, r) in zip(centers, CIRCLES):
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=COLOR, width=STROKE)
    return im


def ease(t):
    # smoothstep: gentle start / stop, continuous motion over full duration
    return t * t * (3 - 2 * t)


def main():
    os.makedirs(FRAMES_DIR, exist_ok=True)
    for f in os.listdir(FRAMES_DIR):
        os.remove(os.path.join(FRAMES_DIR, f))
    for i in range(N_FRAMES):
        t = ease(i / (N_FRAMES - 1))
        centers = []
        for cx, cy, _ in CIRCLES:
            x = int(round(cx + (TARGET[0] - cx) * t))
            y = int(round(cy + (TARGET[1] - cy) * t))
            centers.append((x, y))
        render(centers).save(os.path.join(FRAMES_DIR, f"{i:04d}.png"))

    # sanity: first frame must equal the reference
    ref_path = os.path.join(HERE, "first_frame.png")
    if os.path.exists(ref_path):
        ref = np.array(Image.open(ref_path).convert("RGB"))
        first = np.array(Image.open(os.path.join(FRAMES_DIR, "0000.png")))
        assert (ref == first).all(), "first frame does not match reference"

    out = os.path.join(OUT_DIR, "video.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(FRAMES_DIR, "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), out,
    ], check=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
