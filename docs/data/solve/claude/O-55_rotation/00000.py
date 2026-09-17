#!/usr/bin/env python3
"""Regenerate /app/output/video.mp4: camera orbits the fixed 6-block sculpture
from azimuth 140 deg to 320 deg at constant 31 deg elevation (orthographic view)."""
import os
import subprocess
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import render  # noqa: E402

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

# projection parameters fitted to first_frame.png (exact pixel match)
SCALE, CX, CY = 103.48, 591.275, 655.15
EL = 31.0
AZ0, AZ1 = 140.0, 320.0
N_FRAMES, FPS = 21, 16


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    frames_dir = os.path.join(OUT_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    ref = np.array(Image.open(FIRST).convert("RGB"))
    first = np.array(render(AZ0, EL, SCALE, CX, CY))
    mism = int((first != ref).any(2).sum())
    print(f"first-frame render mismatch vs first_frame.png: {mism} px")

    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)
        az = AZ0 + (AZ1 - AZ0) * t  # linear, evenly paced horizontal rotation
        if i == 0:
            im = Image.fromarray(ref)  # first frame is exactly first_frame.png
        else:
            im = render(az, EL, SCALE, CX, CY)
        im.save(os.path.join(frames_dir, f"frame_{i:03d}.png"))

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(frames_dir, "frame_%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT,
    ]
    subprocess.run(cmd, check=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
