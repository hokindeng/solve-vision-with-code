#!/usr/bin/env python3
"""Fill the empty fifth circle with the next color in the sequence (blue).

The sequence is blue, teal, blue, teal -> the next color is blue.
Only pixels strictly inside the gray placeholder ring are modified; the
ring itself, the other circles and the background stay exactly as in
first_frame.png in every frame.
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
N_FRAMES = 64

# Geometry of the placeholder circle, measured from first_frame.png
CX, CY = 912.0, 512.0        # centre of the fifth circle
INNER_R = 70.5               # inner radius of the gray ring (ring spans 70.5..78.5)
NEXT_COLOR = np.array([43, 104, 223], dtype=np.float32)  # blue, same as circles 1 and 3


def ease_in_out(t: float) -> float:
    """Smoothstep easing."""
    return t * t * (3.0 - 2.0 * t)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    base = np.array(Image.open(FIRST).convert("RGB"))
    h, w, _ = base.shape

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((xx - CX) ** 2 + (yy - CY) ** 2)

    # Only white pixels inside the ring's inner edge may ever change.
    white = np.all(base == 255, axis=2)
    editable = white & (dist < INNER_R + 0.5)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "medium",
        "-movflags", "+faststart",
        OUT,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)
        r = INNER_R * ease_in_out(t)
        frame = base.astype(np.float32).copy()
        if r > 0:
            # Anti-aliased disc: coverage falls off over one pixel at the edge.
            cov = np.clip(r - dist + 0.5, 0.0, 1.0)
            cov[~editable] = 0.0
            cov3 = cov[..., None]
            frame = frame * (1.0 - cov3) + NEXT_COLOR * cov3
        out = np.clip(np.rint(frame), 0, 255).astype(np.uint8)
        proc.stdin.write(out.tobytes())

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit(f"ffmpeg failed with code {proc.returncode}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
