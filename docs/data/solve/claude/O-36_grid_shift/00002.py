#!/usr/bin/env python3
"""Regenerate /app/output/video.mp4: shift every blue block one grid cell downward."""
import os, subprocess
import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS, GRID = 35, 16, 9

def main():
    first = np.array(Image.open(FIRST).convert("RGB"))
    H, W, _ = first.shape
    pitch = H / GRID

    # Blocks = connected components of non-white, non-gridline pixels (black border + blue fill).
    gray = (first == [51, 51, 51]).all(-1)
    white = (first == [255, 255, 255]).all(-1)
    mask = ~(gray | white)
    mask = ndimage.binary_closing(mask, iterations=2)
    lab, n = ndimage.label(mask)
    sprites = []
    background = first.copy()
    for sl in ndimage.find_objects(lab):
        r0, r1 = max(sl[0].start - 2, 0), min(sl[0].stop + 2, H)
        c0, c1 = max(sl[1].start - 2, 0), min(sl[1].stop + 2, W)
        crop = first[r0:r1, c0:c1].copy()
        background[r0:r1, c0:c1] = 255            # erase block (blocks never touch grid lines)
        cy = (r0 + r1) / 2.0
        row = int(cy // pitch)
        assert row + 1 < GRID, "block would leave the grid"
        target_cy = (row + 1.5) * pitch
        dy_total = int(round(target_cy - cy))
        sprites.append((r0, c0, crop, dy_total))
    assert len(sprites) == 10, len(sprites)

    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)
        s = t * t * (3 - 2 * t)                    # smoothstep ease in/out
        frame = background.copy() if i > 0 else first.copy()
        if i > 0:
            for r0, c0, crop, dy_total in sprites:
                dy = int(round(s * dy_total))
                h, w = crop.shape[:2]
                dst = frame[r0 + dy:r0 + dy + h, c0:c0 + w]
                keep = ~(crop == 255).all(-1)      # paste block pixels only, keep grid lines visible
                dst[keep] = crop[keep]
        frames.append(frame)

    raw = np.stack(frames).tobytes()
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT]
    subprocess.run(cmd, input=raw, check=True)
    print("wrote", OUT, len(frames), "frames")

if __name__ == "__main__":
    main()
