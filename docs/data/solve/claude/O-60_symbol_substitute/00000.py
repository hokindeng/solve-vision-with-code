#!/usr/bin/env python3
"""Substitute the solid right triangle at position 2 with a blue solid (upward) triangle.

Animation: the old symbol fades out into white, then the new symbol fades in from white.
Every other pixel is left exactly as in first_frame.png.
"""
import os
import subprocess
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

FPS = 16
N_FRAMES = 52
WHITE = np.array([255, 255, 255], dtype=np.float64)
BLUE = np.array([0, 0, 255], dtype=np.float64)

# Cell 2 interior (border pixels at x=307/403, y=464/560 are excluded).
CELL_X0, CELL_X1, CELL_Y0, CELL_Y1 = 308, 402, 465, 559
# New symbol geometry: same bbox as the old symbol / same size as the reference symbol.
TRI_UP = [(355, 478), (321, 546), (389, 546)]


def ease(t):
    """Smoothstep easing in [0,1]."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def main():
    base = np.array(Image.open(FIRST).convert("RGB")).astype(np.float64)
    H, W, _ = base.shape

    # Old symbol mask: non-white pixels inside cell 2 interior.
    cell = base[CELL_Y0:CELL_Y1 + 1, CELL_X0:CELL_X1 + 1]
    old_mask = np.zeros((H, W), dtype=bool)
    old_mask[CELL_Y0:CELL_Y1 + 1, CELL_X0:CELL_X1 + 1] = (cell != 255).any(axis=2)

    # New symbol mask.
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).polygon(TRI_UP, fill=255)
    new_mask = np.array(m) > 0
    # Keep the new symbol strictly inside the cell interior.
    inside = np.zeros((H, W), dtype=bool)
    inside[CELL_Y0:CELL_Y1 + 1, CELL_X0:CELL_X1 + 1] = True
    new_mask &= inside

    half = N_FRAMES // 2  # frames 0..25 fade out, 26..51 fade in
    frames = []
    for i in range(N_FRAMES):
        img = base.copy()
        if i < half:
            # fade out: t goes 0 (frame 0, original) -> 1 (fully white) at frame half-1
            t = ease(i / (half - 1))
            img[old_mask] = base[old_mask] * (1 - t) + WHITE * t
        else:
            # old symbol gone; fade in new symbol from white to blue
            t = ease((i - half + 1) / (N_FRAMES - half))
            img[old_mask] = WHITE
            img[new_mask] = WHITE * (1 - t) + BLUE * t
        frames.append(np.clip(np.rint(img), 0, 255).astype(np.uint8))

    os.makedirs(OUT_DIR, exist_ok=True)
    frames_dir = os.path.join(OUT_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for i, f in enumerate(frames):
        Image.fromarray(f).save(os.path.join(frames_dir, f"{i:04d}.png"))

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(FPS),
        "-i", os.path.join(frames_dir, "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "0", "-preset", "veryslow",
        "-sws_flags", "+accurate_rnd+full_chroma_int",
        "-r", str(FPS),
        OUT,
    ]
    subprocess.run(cmd, check=True)
    print(f"wrote {OUT} ({len(frames)} frames @ {FPS} fps)")


if __name__ == "__main__":
    main()
