#!/usr/bin/env python3
"""Generate the analogy video: recolor the rectangle (mint -> green), then move it down.

Everything outside the rectangle's own footprint (and its destination) is copied
verbatim from first_frame.png in every frame.
"""
import os
import subprocess
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

FPS = 16
N_FRAMES = 60

MINT = np.array([45, 229, 168], dtype=np.float32)
GREEN = np.array([7, 153, 80], dtype=np.float32)   # color demonstrated by top row (B, C)
DY = 288 - 263                                       # circle B -> C vertical shift (25 px)

# Rectangle (incl. black outline) bounding box in first frame
X0, X1, Y0, Y1 = 83, 307 + 1, 626, 738 + 1

# Timeline (frame indices)
RECOLOR_START, RECOLOR_END = 2, 27
MOVE_START, MOVE_END = 32, 56


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0.0, 1.0))


def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    patch = base[Y0:Y1, X0:X1].astype(np.float32)
    fill_mask = np.abs(patch - MINT).sum(axis=2) < 30  # interior pixels to recolor

    # Background with the rectangle removed (pure white underneath)
    bg = base.copy()
    bg[Y0:Y1, X0:X1] = 255

    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for i in range(N_FRAMES):
        c = ease((i - RECOLOR_START) / (RECOLOR_END - RECOLOR_START))
        m = ease((i - MOVE_START) / (MOVE_END - MOVE_START))

        p = patch.copy()
        col = MINT * (1 - c) + GREEN * c
        p[fill_mask] = col
        p = np.clip(np.rint(p), 0, 255).astype(np.uint8)

        dy = int(round(m * DY))
        frame = bg.copy() if i > 0 else base.copy()
        if i > 0:
            frame[Y0 + dy:Y1 + dy, X0:X1] = p
        frames.append(frame)

    # Encode with ffmpeg (H.264, yuv420p)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1024x1024", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print("wrote", OUT, len(frames), "frames")


if __name__ == "__main__":
    main()
