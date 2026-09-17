#!/usr/bin/env python3
"""Move the blue triangular agent from the green start node to the red end node
along the shortest directed path (green -> bottom white node -> red)."""
import subprocess
import numpy as np
from PIL import Image

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS = 16
N_FRAMES = 30

# Node centres measured from the first frame (x, y)
GREEN = (804, 466)
BOTTOM = (474, 752)
RED = (584, 214)
# Shortest path respecting edge directions: green -> bottom -> red (2 steps).
PATH = [GREEN, BOTTOM, RED]

GREEN_FILL = np.array([0, 128, 0], dtype=np.uint8)


def main():
    frame0 = np.array(Image.open(SRC).convert("RGB"))
    h, w, _ = frame0.shape

    # Extract the agent sprite: blue (0,0,255) and dark blue (0,0,139) pixels.
    blue = (frame0[..., 0] == 0) & (frame0[..., 1] == 0) & (frame0[..., 2] >= 139)
    ys, xs = np.nonzero(blue)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    sprite = frame0[y0:y1, x0:x1].copy()
    sprite_mask = blue[y0:y1, x0:x1]
    # Sprite offset relative to the green node centre
    off_x, off_y = x0 - GREEN[0], y0 - GREEN[1]

    # Background: first frame with the agent removed (restore green fill).
    background = frame0.copy()
    background[blue] = GREEN_FILL

    def ease(t):
        return t * t * (3 - 2 * t)  # smoothstep

    def agent_pos(i):
        # Segment A: frames 0..14 (green -> bottom); Segment B: frames 15..29 (bottom -> red)
        if i <= 14:
            t = ease(i / 14.0)
            a, b = PATH[0], PATH[1]
        else:
            t = ease((i - 15) / 14.0)
            a, b = PATH[1], PATH[2]
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)

    frames = []
    for i in range(N_FRAMES):
        if i == 0:
            frames.append(frame0.copy())
            continue
        cx, cy = agent_pos(i)
        px = int(round(cx)) + off_x
        py = int(round(cy)) + off_y
        f = background.copy()
        sh, sw = sprite_mask.shape
        region = f[py:py + sh, px:px + sw]
        region[sprite_mask] = sprite[sprite_mask]
        frames.append(f)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {OUT} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
