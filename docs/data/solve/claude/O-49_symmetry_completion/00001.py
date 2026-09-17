"""Complete a left/right mirror-symmetric grid pattern by filling in the
missing cells on the right side, animated over 35 frames."""
import os, subprocess, shutil
import numpy as np
from PIL import Image

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS, N_FRAMES = 16, 35
BLUE = np.array([37, 99, 235], np.uint8)
GRID = np.array([203, 213, 225], np.uint8)


def runs(mask):
    """Return (start, end) inclusive runs of True in a 1-D mask."""
    idx = np.where(mask)[0]
    out, s = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b != a + 1:
            out.append((s, a)); s = b
    out.append((s, idx[-1]))
    return out


def main():
    base = np.array(Image.open(SRC).convert("RGB"))
    H, W, _ = base.shape
    grid = (base == GRID).all(-1)
    ys, xs = np.where(grid)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    # grid lines span the whole grid, so project the mask onto each axis
    vx = runs(grid.sum(axis=0) > 0.5 * (y1 - y0))   # vertical line x-extents
    hy = runs(grid.sum(axis=1) > 0.5 * (x1 - x0))   # horizontal line y-extents
    # cell interiors between consecutive lines
    col_ext = [(vx[i][1] + 1, vx[i + 1][0] - 1) for i in range(len(vx) - 1)]
    row_ext = [(hy[i][1] + 1, hy[i + 1][0] - 1) for i in range(len(hy) - 1)]
    nc, nr = len(col_ext), len(row_ext)

    def is_blue(r, c):
        ya, yb = row_ext[r]; xa, xb = col_ext[c]
        return (base[ya:yb + 1, xa:xb + 1] == BLUE).all(-1).mean() > 0.5

    # missing cells: right-side cells whose mirror on the left is filled
    missing = []
    for r in range(nr):
        for c in range(nc // 2, nc):
            m = nc - 1 - c
            if is_blue(r, m) and not is_blue(r, c):
                missing.append((r, c))
    missing.sort()  # top-to-bottom, left-to-right reading order
    print("missing cells:", missing)

    # schedule: each cell grows from its centre over `grow` frames, staggered
    # so the last one finishes exactly on the final frame.
    grow = 8
    last_start = N_FRAMES - 1 - grow
    first_start = 1
    starts = np.linspace(first_start, last_start, len(missing)) if len(missing) > 1 else [last_start]

    frames_dir = "/app/output/frames"
    shutil.rmtree(frames_dir, ignore_errors=True)
    os.makedirs(frames_dir)
    for f in range(N_FRAMES):
        img = base.copy()
        for (r, c), s in zip(missing, starts):
            t = np.clip((f - s) / grow, 0.0, 1.0)
            if t <= 0:
                continue
            t = 1 - (1 - t) ** 3  # ease-out
            ya, yb = row_ext[r]; xa, xb = col_ext[c]
            h, w = yb - ya + 1, xb - xa + 1
            hh, ww = max(1, int(round(h * t))), max(1, int(round(w * t)))
            oy, ox = ya + (h - hh) // 2, xa + (w - ww) // 2
            img[oy:oy + hh, ox:ox + ww] = BLUE
        Image.fromarray(img).save(f"{frames_dir}/{f:04d}.png")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", f"{frames_dir}/%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "12", "-preset", "slow", OUT], check=True)
    shutil.rmtree(frames_dir)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
