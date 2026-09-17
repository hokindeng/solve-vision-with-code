#!/usr/bin/env python3
"""Replace the solid right triangle at position 3 with a violet hollow star.

Old symbol fades out to white, then new symbol fades in from white.
"""
import os
import subprocess
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

N_FRAMES = 52
FPS = 16

# Cell 3 (border at x=306/402, y=464/560); interior excludes the border.
CELL_X0, CELL_X1 = 307, 402   # exclusive end
CELL_Y0, CELL_Y1 = 465, 560
CELL_CENTER = (354, 512)

# Reference panel (border at x=887/1006, y=18/137); interior excludes border.
REF_X0, REF_X1 = 888, 1006
REF_Y0, REF_Y1 = 19, 137
REF_STAR_CENTER = (946, 77)   # measured centre of the reference star


def smoothstep(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def main():
    base = np.array(Image.open(FIRST).convert("RGB")).astype(np.float32)
    H, W, _ = base.shape
    white = np.full_like(base, 255.0)

    # --- old symbol: everything non-white inside cell 3's interior
    cell = base[CELL_Y0:CELL_Y1, CELL_X0:CELL_X1]
    old_layer = cell.copy()  # colours of the old symbol over white

    # --- new symbol: copy reference star pixels, translated to the cell centre
    ref = base[REF_Y0:REF_Y1, REF_X0:REF_X1]
    ref_mask = np.abs(ref - 255).sum(axis=2) > 0
    dx = CELL_CENTER[0] - REF_STAR_CENTER[0]
    dy = CELL_CENTER[1] - REF_STAR_CENTER[1]
    new_layer = np.full_like(cell, 255.0)
    ys, xs = np.nonzero(ref_mask)
    for y, x in zip(ys, xs):
        gy = y + REF_Y0 + dy - CELL_Y0
        gx = x + REF_X0 + dx - CELL_X0
        if 0 <= gy < new_layer.shape[0] and 0 <= gx < new_layer.shape[1]:
            new_layer[gy, gx] = ref[y, x]

    # --- timeline: frames 0..25 fade out (frame 25 fully white),
    #               frames 26..51 fade in (frame 51 fully coloured)
    half = 25
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for f in range(N_FRAMES):
        frame = base.copy()
        if f <= half:
            a = 1.0 - smoothstep(f / half)
            layer = old_layer
        else:
            a = smoothstep((f - half) / (N_FRAMES - 1 - half))
            layer = new_layer
        region = 255.0 + (layer - 255.0) * a
        frame[CELL_Y0:CELL_Y1, CELL_X0:CELL_X1] = region
        frames.append(np.clip(np.rint(frame), 0, 255).astype(np.uint8))

    # Sanity: first frame is exactly the input.
    assert np.array_equal(frames[0], base.astype(np.uint8))

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        proc.stdin.write(fr.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print("wrote", OUT, "frames:", len(frames))


if __name__ == "__main__":
    main()
