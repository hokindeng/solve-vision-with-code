#!/usr/bin/env python3
"""Draw a red circle around the arrow that points the wrong way.

All arrows run clockwise except the one on the lower right, which points
up-right (counter-clockwise). The circle is drawn progressively as an arc
sweeping 360 degrees over the video duration; frame 0 is the untouched
first frame and every other pixel stays unchanged.
"""
import subprocess
import numpy as np
from PIL import Image, ImageDraw

W = H = 1024
FPS = 16
N_FRAMES = 48
SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"

# Odd arrow bounding box (measured): x 765..858, y 585..740
CX, CY = 812, 662
R = 100
LINE_W = 6
RED = (255, 0, 0)


def frame(i, base):
    im = base.copy()
    if i == 0:
        return im
    t = i / (N_FRAMES - 1)
    sweep = 360.0 * t
    d = ImageDraw.Draw(im)
    bbox = [CX - R, CY - R, CX + R, CY + R]
    start = -90.0
    if sweep >= 359.9:
        d.ellipse(bbox, outline=RED, width=LINE_W)
    else:
        d.arc(bbox, start=start, end=start + sweep, fill=RED, width=LINE_W)
    return im


def main():
    base = Image.open(SRC).convert("RGB")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(N_FRAMES):
        p.stdin.write(np.asarray(frame(i, base), dtype=np.uint8).tobytes())
    p.stdin.close()
    p.wait()
    assert p.returncode == 0, "ffmpeg failed"
    print("wrote", OUT)


if __name__ == "__main__":
    main()
