#!/usr/bin/env python3
"""Regenerate /app/output/video.mp4: 27 purple blocks on a 9x9 grid shift right by 2 cells."""
import os, subprocess, shutil, tempfile
import numpy as np
from PIL import Image

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT = os.path.join(APP, "output", "video.mp4")
N, W, FPS, FRAMES, SHIFT = 9, 1024, 16, 35, 2
PURPLE, BLACK, GRID, WHITE = (128, 0, 128), (0, 0, 0), (51, 51, 51), (255, 255, 255)
INSET, OUTLINE = 9, 2          # block spans cell+9 .. cell+104 (96px), 2px black outline
EDGES = [int(i * W / N) for i in range(N + 1)]

def main():
    src = np.array(Image.open(FIRST).convert("RGB"))
    E = EDGES[:-1] + [W - 1]          # block bounds use the last pixel column/row as the final edge

    # Locate occupied cells.
    cells = []
    for r in range(N):
        for c in range(N):
            cy, cx = (EDGES[r] + EDGES[r + 1]) // 2, (EDGES[c] + EDGES[c + 1]) // 2
            if tuple(src[cy, cx]) == PURPLE:
                cells.append((r, c))
    assert len(cells) == 27, len(cells)
    assert max(c for _, c in cells) + SHIFT < N, "blocks must stay inside the grid"

    def rect(r, c):
        # inclusive block rectangle for grid cell (r, c)
        return E[r] + INSET, E[r + 1] - INSET, E[c] + INSET, E[c + 1] - INSET

    # Background: the first frame with every block erased (white + restored grid lines).
    bg = src.copy()
    for r, c in cells:
        y0, y1, x0, x1 = rect(r, c)
        bg[y0:y1 + 1, x0:x1 + 1] = WHITE
    for e in EDGES[:-1]:
        bg[e, :] = GRID
        bg[:, e] = GRID

    def draw(frame, y0, y1, x0, x1):
        frame[y0:y1 + 1, x0:x1 + 1] = BLACK
        frame[y0 + OUTLINE:y1 + 1 - OUTLINE, x0 + OUTLINE:x1 + 1 - OUTLINE] = PURPLE

    def ease(t):
        return 0.5 - 0.5 * np.cos(np.pi * t)

    # Two discrete one-cell steps, each eased, filling the whole duration.
    def progress(i):
        if i == FRAMES - 1:
            return float(SHIFT)
        t = i / (FRAMES - 1) * SHIFT
        step = min(int(t), SHIFT - 1)
        return step + ease(t - step)

    tmp = tempfile.mkdtemp()
    for i in range(FRAMES):
        p = progress(i)
        k = min(int(np.floor(p)), SHIFT); f = p - k
        frame = bg.copy()
        for r, c in cells:
            y0, y1, xa0, xa1 = rect(r, c + k)
            if f > 0:
                _, _, xb0, xb1 = rect(r, c + k + 1)
                x0 = int(round(xa0 + f * (xb0 - xa0)))
                x1 = int(round(xa1 + f * (xb1 - xa1)))
            else:
                x0, x1 = xa0, xa1
            draw(frame, y0, y1, x0, x1)
        if i == 0:
            assert (frame == src).all(), "frame 0 must reproduce first_frame.png"
        Image.fromarray(frame).save(os.path.join(tmp, f"f{i:04d}.png"))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", os.path.join(tmp, "f%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", OUT], check=True)
    shutil.rmtree(tmp)
    print("wrote", OUT)

if __name__ == "__main__":
    main()
