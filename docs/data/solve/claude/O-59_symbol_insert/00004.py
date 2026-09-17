#!/usr/bin/env python3
"""Insert a blue solid star at position 5.

Slot 5 is empty in the first frame, so no existing symbol needs to slide.
The star (copied pixel-exactly from the reference panel) fades in above the
gap and then slides down into the target slot. Every other pixel is left
exactly as in first_frame.png.
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
N_FRAMES = 24

# Geometry measured from first_frame.png
STAR_BBOX = (40, 114, 907, 977)          # y0, y1, x0, x1 of star pixels in reference panel
STAR_CENTROID = (76.49, 945.57)          # (y, x)
SLOT5_CENTER = (512, 722)                # (y, x) of the empty target slot
RISE = 110                               # pixels above the slot center where the star appears

# Timeline (frame indices)
FADE_START, FADE_END = 1, 9              # fade in while hovering above the gap
SLIDE_START, SLIDE_END = 9, 22           # slide down into the slot; hold afterwards


def ease(t):
    """Smooth ease-in-out."""
    return t * t * (3 - 2 * t)


def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    H, W, _ = base.shape

    y0, y1, x0, x1 = STAR_BBOX
    patch = base[y0:y1, x0:x1].astype(np.float32)
    mask = ((patch[..., 0] < 128) & (patch[..., 2] > 128)).astype(np.float32)  # blue star pixels
    ph, pw = mask.shape
    # Top-left offset so that the star centroid lands on a given center
    off_y = y0 - STAR_CENTROID[0]
    off_x = x0 - STAR_CENTROID[1]

    frames = []
    for i in range(N_FRAMES):
        frame = base.astype(np.float32).copy()

        if i >= FADE_START:
            # opacity
            if i >= FADE_END:
                alpha = 1.0
            else:
                alpha = (i - FADE_START) / (FADE_END - FADE_START)
            # vertical position
            if i <= SLIDE_START:
                dy = -RISE
            elif i >= SLIDE_END:
                dy = 0
            else:
                t = ease((i - SLIDE_START) / (SLIDE_END - SLIDE_START))
                dy = -RISE * (1 - t)
            cy = SLOT5_CENTER[0] + dy
            cx = SLOT5_CENTER[1]
            ty = int(round(cy + off_y))
            tx = int(round(cx + off_x))

            region = frame[ty:ty + ph, tx:tx + pw]
            a = (mask * alpha)[..., None]
            region[:] = region * (1 - a) + patch * a

        frames.append(np.clip(np.round(frame), 0, 255).astype(np.uint8))

    # First frame must be identical to the input
    assert np.array_equal(frames[0], base)

    os.makedirs(OUT_DIR, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {OUT} ({len(frames)} frames @ {FPS} fps)")


if __name__ == "__main__":
    main()
