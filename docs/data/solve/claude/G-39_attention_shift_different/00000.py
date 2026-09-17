#!/usr/bin/env python3
"""Move the green attention box from the left object to the right object."""
import subprocess
import numpy as np
from PIL import Image

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
N_FRAMES, FPS = 25, 16
GREEN = np.array([70, 140, 70], dtype=np.uint8)
DX, DY = 502, 0  # right-object centre minus left-object centre (measured)


def main():
    first = np.array(Image.open(SRC).convert("RGB"))
    box = (first == GREEN).all(axis=2)
    ys, xs = np.where(box)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    patch = box[y0:y1, x0:x1]

    # Background = frame with the box removed. Box sits on plain white.
    bg = first.copy()
    bg[box] = 255

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{first.shape[1]}x{first.shape[0]}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)
        s = t * t * (3 - 2 * t)  # smoothstep ease-in-out
        dx, dy = int(round(DX * s)), int(round(DY * s))
        frame = first.copy() if i == 0 else bg.copy()
        if i > 0:
            frame[y0 + dy:y1 + dy, x0 + dx:x1 + dx][patch] = GREEN
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    assert proc.returncode == 0


if __name__ == "__main__":
    main()
