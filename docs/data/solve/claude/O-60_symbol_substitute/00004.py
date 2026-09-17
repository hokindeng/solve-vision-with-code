#!/usr/bin/env python3
"""Substitute the solid heart in cell 3 with a blue solid diamond.

Old symbol fades out to white, then the new symbol fades in from white,
at the same position. Every other pixel is left untouched.
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
N_FRAMES = 52

# Cell 3 geometry (measured from first_frame.png): box border x 412..508, y 464..560
CELL_X0, CELL_X1, CELL_Y0, CELL_Y1 = 412, 508, 464, 560
CELL_CX, CELL_CY = (CELL_X0 + CELL_X1) // 2, (CELL_Y0 + CELL_Y1) // 2  # (460, 512)

HEART_COLOR = np.array([238, 130, 238], dtype=np.float32)
DIAMOND_COLOR = np.array([0, 0, 255], dtype=np.float32)   # same blue as reference panel
DIAMOND_HALF_DIAG = 34                                     # same size as reference diamond
WHITE = np.array([255, 255, 255], dtype=np.float32)


def smoothstep(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def main():
    base = np.array(Image.open(FIRST).convert("RGB")).astype(np.float32)
    H, W, _ = base.shape

    # Mask of the old heart: solid pink pixels inside cell 3 (interior only, not border).
    region = np.zeros((H, W), dtype=bool)
    region[CELL_Y0 + 1:CELL_Y1, CELL_X0 + 1:CELL_X1] = True
    heart_mask = region & np.all(base == HEART_COLOR, axis=-1)

    # Mask of the new diamond: |dx| + |dy| <= half-diagonal, centred in the cell.
    yy, xx = np.mgrid[0:H, 0:W]
    diamond_mask = (np.abs(xx - CELL_CX) + np.abs(yy - CELL_CY)) <= DIAMOND_HALF_DIAG
    diamond_mask &= region

    # Timeline: heart fades out over first half, diamond fades in over second half.
    half = N_FRAMES // 2  # 26 frames each
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for i in range(N_FRAMES):
        if i < half:
            a_heart = 1.0 - smoothstep(i / (half - 1))   # frame 0 -> 1.0, frame 25 -> 0.0
            a_dia = 0.0
        else:
            a_heart = 0.0
            a_dia = smoothstep((i - half) / (N_FRAMES - 1 - half))  # 0.0 -> 1.0 at last frame
        f = base.copy()
        f[heart_mask] = WHITE + a_heart * (HEART_COLOR - WHITE)
        f[diamond_mask] = (1.0 - a_dia) * f[diamond_mask] + a_dia * DIAMOND_COLOR
        frames.append(np.clip(np.round(f), 0, 255).astype(np.uint8))

    # Encode with ffmpeg: H.264, yuv420p, 16 fps.
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-preset", "slow", "-crf", "12",
        "-pix_fmt", "yuv420p", "-r", str(FPS), "-movflags", "+faststart",
        OUT,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        raise SystemExit("ffmpeg failed")
    # Also save first/last frame PNGs for inspection.
    Image.fromarray(frames[0]).save(os.path.join(OUT_DIR, "frame_first.png"))
    Image.fromarray(frames[-1]).save(os.path.join(OUT_DIR, "frame_last.png"))
    print(f"wrote {OUT} ({len(frames)} frames @ {FPS} fps)")


if __name__ == "__main__":
    main()
