#!/usr/bin/env python3
"""Animate a green rectangular highlight box around the bathroom in the floorplan.

The bathroom is the top-left room (toilet, sink, bathtub). Its walls span
x=55..414 and y=56..400 in first_frame.png. The box is traced clockwise along the
perimeter just outside the walls over 28 frames; all other pixels are unchanged.
"""
import subprocess
import numpy as np
from PIL import Image, ImageDraw

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS = 16
N_FRAMES = 28

# Box just outside the bathroom walls
X0, Y0, X1, Y1 = 49, 15, 420, 406
THICK = 5
GREEN = (0, 200, 0)

corners = [(X0, Y0), (X1, Y0), (X1, Y1), (X0, Y1), (X0, Y0)]
seg_len = [abs(corners[i + 1][0] - corners[i][0]) + abs(corners[i + 1][1] - corners[i][1])
           for i in range(4)]
perim = sum(seg_len)


def point_at(d):
    """Point along the perimeter at distance d from the top-left corner (clockwise)."""
    for i in range(4):
        if d <= seg_len[i]:
            (ax, ay), (bx, by) = corners[i], corners[i + 1]
            t = d / seg_len[i]
            return (ax + (bx - ax) * t, ay + (by - ay) * t)
        d -= seg_len[i]
    return corners[-1]


def make_frame(base, progress):
    img = base.copy()
    if progress <= 0:
        return img
    draw = ImageDraw.Draw(img)
    d_total = perim * progress
    d = 0.0
    for i in range(4):
        if d_total <= d:
            break
        (ax, ay) = corners[i]
        end_d = min(d_total, d + seg_len[i])
        (bx, by) = point_at(end_d)
        # axis-aligned thick segment drawn as a filled rectangle, so corners are crisp
        h = THICK // 2
        xa, xb = sorted([ax, bx]); ya, yb = sorted([ay, by])
        draw.rectangle([xa - h, ya - h, xb + h, yb + h], fill=GREEN)
        d += seg_len[i]
    return img


def main():
    base = Image.open(SRC).convert("RGB")
    frames = []
    for k in range(N_FRAMES):
        # first frame untouched; box fully drawn by the last frame, eased slightly
        t = k / (N_FRAMES - 1)
        progress = 0.0 if k == 0 else (1 - (1 - t) ** 2)
        frames.append(np.array(make_frame(base, progress)))
    frames[-1] = np.array(make_frame(base, 1.0))

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "1024x1024", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "1", "-preset", "slow", "-tune", "stillimage", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.astype(np.uint8).tobytes())
    p.stdin.close()
    p.wait()
    assert p.returncode == 0, "ffmpeg failed"
    print("wrote", OUT)


if __name__ == "__main__":
    main()
