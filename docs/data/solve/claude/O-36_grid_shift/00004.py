#!/usr/bin/env python3
"""Regenerate /app/output/video.mp4: all lime blocks shift up one grid cell."""
import os, subprocess, shutil, tempfile
import numpy as np
from PIL import Image

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT = os.path.join(APP, "output", "video.mp4")
CELL, N, FPS, FRAMES = 64, 16, 16, 35
LIME = np.array([0, 255, 0], np.uint8)

def main():
    im = np.array(Image.open(FIRST).convert("RGB"))
    H, W, _ = im.shape

    # Locate blocks (cell whose centre is lime) and capture the block sprite.
    cells = [(r, c) for r in range(N) for c in range(N)
             if np.array_equal(im[r*CELL + CELL//2, c*CELL + CELL//2], LIME)]
    r0, c0 = cells[0]
    cell = im[r0*CELL:(r0+1)*CELL, c0*CELL:(c0+1)*CELL]
    nonwhite = ~np.all(cell == 255, axis=2)
    # exclude the grid line pixels on the cell's top/left edge
    nonwhite[0, :] = False; nonwhite[:, 0] = False
    ys, xs = np.where(nonwhite)
    y0, y1, x0, x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
    sprite = cell[y0:y1, x0:x1].copy()

    # Background: the frame with every block erased (interior is plain white).
    bg = im.copy()
    for r, c in cells:
        bg[r*CELL+y0:r*CELL+y1, c*CELL+x0:c*CELL+x1] = 255

    def ease(t):  # smoothstep for gentle start/stop
        return t*t*(3 - 2*t)

    tmp = tempfile.mkdtemp()
    for i in range(FRAMES):
        t = i / (FRAMES - 1)
        dy = int(round(CELL * ease(t)))
        f = bg.copy()
        for r, c in cells:
            y = r*CELL + y0 - dy
            f[y:y+(y1-y0), c*CELL+x0:c*CELL+x1] = sprite
        Image.fromarray(f).save(os.path.join(tmp, f"f{i:04d}.png"))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", os.path.join(tmp, "f%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
                    "-preset", "slow", "-r", str(FPS), OUT], check=True)
    shutil.rmtree(tmp)
    print(f"wrote {OUT}: {len(cells)} blocks, {FRAMES} frames")

if __name__ == "__main__":
    main()
