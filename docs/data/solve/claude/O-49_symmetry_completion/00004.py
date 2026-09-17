#!/usr/bin/env python3
"""Complete the striped pattern by mirroring the left half to the right half."""
import subprocess, os
import numpy as np
from PIL import Image

SRC = "/app/first_frame.png"
OUT_DIR = "/app/output"
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 35

GRID_COL = np.array([203, 213, 225], np.uint8)
FILL_COL = np.array([217, 119, 6], np.uint8)

def cell_spans(gridmask_1d):
    """Given a 1D boolean array (True on grid lines), return interior spans between lines."""
    idx = np.where(gridmask_1d)[0]
    groups, cur = [], [idx[0]]
    for v in idx[1:]:
        if v == cur[-1] + 1:
            cur.append(v)
        else:
            groups.append(cur); cur = [v]
    groups.append(cur)
    return [(groups[i][-1] + 1, groups[i + 1][0]) for i in range(len(groups) - 1)]

def main():
    base = np.array(Image.open(SRC).convert("RGB"))
    g = np.all(base == GRID_COL, axis=2)
    rows = cell_spans(g.sum(1) > 500)
    cols = cell_spans(g.sum(0) > 500)
    n = len(cols)

    def filled(r, c):
        y0, y1 = rows[r]; x0, x1 = cols[c]
        return np.all(base[y0:y1, x0:x1] == FILL_COL)

    # cells to fill: right-half cells whose mirror on the left is filled but they are not
    targets = []
    for r in range(len(rows)):
        for c in range(n // 2, n):
            if filled(r, n - 1 - c) and not filled(r, c):
                targets.append((r, c))

    # stagger fade-in of each cell across the duration
    fade = 8
    span = N_FRAMES - 2 - fade  # frames available for starts (frame 0 untouched, last frame held)
    starts = [1 + round(i * span / max(1, len(targets) - 1)) for i in range(len(targets))] if len(targets) > 1 else [1]

    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = os.path.join(OUT_DIR, "frames")
    os.makedirs(tmp, exist_ok=True)
    for f in range(N_FRAMES):
        fr = base.copy()
        for (r, c), s in zip(targets, starts):
            a = np.clip((f - s + 1) / fade, 0, 1)
            if f == N_FRAMES - 1:
                a = 1.0
            if a <= 0:
                continue
            y0, y1 = rows[r]; x0, x1 = cols[c]
            region = fr[y0:y1, x0:x1].astype(np.float32)
            fr[y0:y1, x0:x1] = np.round(region * (1 - a) + FILL_COL.astype(np.float32) * a).astype(np.uint8)
        Image.fromarray(fr).save(os.path.join(tmp, f"{f:03d}.png"))

    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", os.path.join(tmp, "%03d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-r", str(FPS), OUT], check=True)
    for fn in os.listdir(tmp):
        os.remove(os.path.join(tmp, fn))
    os.rmdir(tmp)
    print("wrote", OUT, "targets:", targets)

if __name__ == "__main__":
    main()
