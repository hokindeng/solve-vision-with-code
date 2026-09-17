#!/usr/bin/env python3
"""Subtractive (multiply) pigment mixing animation.

Reads /app/first_frame.png, computes the multiplied mix of the two pigments,
and progressively fills the interior of the black-bordered mixing zone with
that color (an ease-in-out square wipe growing from the centre). All other
pixels stay identical to the first frame in every frame.
"""
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

APP = Path("/app")
OUT = APP / "output" / "video.mp4"
FPS = 16
N_FRAMES = 44

LEFT = np.array([120, 25, 0], dtype=np.int64)
RIGHT = np.array([149, 163, 214], dtype=np.int64)


def multiply_mix(c1, c2):
    return np.round(c1 * c2 / 255.0).astype(np.uint8)


def smoothstep(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def main():
    base = np.array(Image.open(APP / "first_frame.png").convert("RGB"))
    h, w, _ = base.shape

    # Locate the black-bordered square (all pure-black pixels belong to it).
    black = (base == 0).all(axis=2)
    ys, xs = np.where(black)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()

    # Interior = white pixels enclosed by the border's bounding box.
    interior = np.zeros((h, w), dtype=bool)
    interior[y0:y1 + 1, x0:x1 + 1] = (base[y0:y1 + 1, x0:x1 + 1] == 255).all(axis=2)
    iy, ix = np.where(interior)
    cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
    half = max(cy - y0, cx - x0) + 1.0

    mixed = multiply_mix(LEFT, RIGHT)
    print(f"mixed colour = {tuple(int(v) for v in mixed)}")

    # Chebyshev distance of each interior pixel from the centre -> square wipe.
    dist = np.maximum(np.abs(iy - cy), np.abs(ix - cx))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), str(OUT),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    hold_start, hold_end = 2, 4  # frames held at start / end
    active = N_FRAMES - hold_start - hold_end
    for i in range(N_FRAMES):
        frame = base.copy()
        if i >= hold_start:
            t = smoothstep((i - hold_start + 1) / active)
            sel = dist <= t * half
            frame[iy[sel], ix[sel]] = mixed
        proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
