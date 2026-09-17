#!/usr/bin/env python3
"""Move two ring-shaped circles so they become concentric at the image center.

Geometry measured from /app/first_frame.png (verified pixel-exact):
  green   ring: center (394, 307), outer radius 380, line width 8, color (210,245,60)
  magenta ring: center (847, 716), outer radius 163, line width 8, color (240,50,230)
Background is pure white. Frame size 1024x1024.
"""
import os
import subprocess
import numpy as np
from PIL import Image, ImageDraw

W = H = 1024
FPS = 16
N_FRAMES = 40
BG = (255, 255, 255)
TARGET = (W / 2, H / 2)  # (512, 512)

# (color, start_center, outer_radius, line_width) -- drawn in this order
RINGS = [
    ((210, 245, 60), (394.0, 307.0), 380, 8),
    ((240, 50, 230), (847.0, 716.0), 163, 8),
]

OUT_DIR = "/app/output"
OUT_MP4 = os.path.join(OUT_DIR, "video.mp4")


def ease(t: float) -> float:
    """Smoothstep ease-in-out on [0,1]."""
    return t * t * (3.0 - 2.0 * t)


def render(centers):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    for (color, _, r, w), (cx, cy) in zip(RINGS, centers):
        cx, cy = int(round(cx)), int(round(cy))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=w)
    return img


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    frames_dir = os.path.join(OUT_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for f in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, f))

    for i in range(N_FRAMES):
        t = ease(i / (N_FRAMES - 1))
        centers = [
            (sx + (TARGET[0] - sx) * t, sy + (TARGET[1] - sy) * t)
            for (_, (sx, sy), _, _) in RINGS
        ]
        render(centers).save(os.path.join(frames_dir, f"frame_{i:03d}.png"))

    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(FPS),
            "-i", os.path.join(frames_dir, "frame_%03d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "12", "-preset", "slow",
            "-r", str(FPS),
            OUT_MP4,
        ],
        check=True,
    )
    print(f"wrote {OUT_MP4}")


if __name__ == "__main__":
    main()
