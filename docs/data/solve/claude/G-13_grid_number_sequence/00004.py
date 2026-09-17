#!/usr/bin/env python3
"""Animate the orange agent from the green start cell through the yellow
numbered waypoints (1 -> 2 -> 3 ...) to the red goal, using shortest
4-connected (Manhattan) paths between consecutive waypoints."""
import subprocess
import numpy as np
from PIL import Image

FIRST = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS = 16
N_FRAMES = 92
GRID_N = 10

ORANGE = np.array([255, 165, 0])
GREEN = np.array([0, 255, 0])
RED = np.array([255, 0, 0])
YELLOW = np.array([255, 255, 0])
GRAY = np.array([150, 150, 150])


def color_mask(img, c):
    return (img == c).all(axis=2)


def group_lines(pos):
    groups, cur = [], [pos[0]]
    for p in pos[1:]:
        if p - cur[-1] <= 2:
            cur.append(p)
        else:
            groups.append(cur)
            cur = [p]
    groups.append(cur)
    return [np.mean(g) for g in groups]


def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    H, W, _ = base.shape

    # --- grid geometry from the gray lines ---
    g = color_mask(base, GRAY)
    xs = group_lines(np.where(g.sum(0) > H * 0.8)[0])
    ys = group_lines(np.where(g.sum(1) > W * 0.8)[0])
    assert len(xs) == GRID_N + 1 and len(ys) == GRID_N + 1, (len(xs), len(ys))

    def cell_center(col, row):
        return int((xs[col] + xs[col + 1]) / 2), int((ys[row] + ys[row + 1]) / 2)

    def cell_of(x, y):
        col = int(np.searchsorted(xs, x) - 1)
        row = int(np.searchsorted(ys, y) - 1)
        return col, row

    # --- agent sprite (orange disc), start cell, goal cell ---
    om = color_mask(base, ORANGE)
    oy, ox = np.where(om)
    x0, x1, y0, y1 = ox.min(), ox.max(), oy.min(), oy.max()
    sprite_mask = om[y0:y1 + 1, x0:x1 + 1]
    acx, acy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    start = cell_of(acx, acy)

    ry, rx = np.where(color_mask(base, RED))
    goal = cell_of(rx.mean(), ry.mean())

    # background: first frame with the agent removed (green restored beneath)
    bg = base.copy()
    bg[om] = GREEN

    # --- yellow waypoints, ordered by their printed number ---
    from scipy import ndimage
    ym = color_mask(base, YELLOW)
    lab, n = ndimage.label(ym)
    waypoints = []
    for i in range(1, n + 1):
        yy, xx = np.where(lab == i)
        col, row = cell_of(xx.mean(), yy.mean())
        cx, cy = cell_center(col, row)
        # count dark digit pixels in the cell to identify the number
        cell = base[int(ys[row]) + 2:int(ys[row + 1]) - 1, int(xs[col]) + 2:int(xs[col + 1]) - 1]
        dark = (cell.sum(axis=2) < 200)
        waypoints.append((col, row, dark))
    waypoints = order_by_digit(waypoints)

    # --- shortest Manhattan path (horizontal, then vertical between waypoints) ---
    stops = [start] + [(c, r) for c, r, _ in waypoints] + [goal]
    path = [start]
    for (c0, r0), (c1, r1) in zip(stops, stops[1:]):
        c, r = c0, r0
        while c != c1:
            c += 1 if c1 > c else -1
            path.append((c, r))
        while r != r1:
            r += 1 if r1 > r else -1
            path.append((c, r))
    n_moves = len(path) - 1
    pts = np.array([cell_center(c, r) for c, r in path], dtype=float)
    # anchor precisely on the sprite's actual start position
    pts[0] = (acx, acy)

    hold_start, hold_end = 4, 4
    move_frames = N_FRAMES - hold_start - hold_end

    def agent_pos(frame):
        t = (frame - hold_start) / move_frames
        t = min(max(t, 0.0), 1.0)
        s = t * n_moves
        i = min(int(np.floor(s)), n_moves - 1)
        f = s - i
        return pts[i] * (1 - f) + pts[i + 1] * f

    sh, sw = sprite_mask.shape
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for k in range(N_FRAMES):
        frame = bg.copy()
        px, py = agent_pos(k)
        tx = int(round(px - sw / 2.0 + 0.5))
        ty = int(round(py - sh / 2.0 + 0.5))
        if k == 0:
            tx, ty = x0, y0
        region = frame[ty:ty + sh, tx:tx + sw]
        region[sprite_mask] = ORANGE
        proc.stdin.write(frame.astype(np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"path: {stops} ({n_moves} moves) -> {OUT}")


def order_by_digit(wps):
    """Sort waypoints by recognising the digit rendered in each cell via
    template matching against DejaVu Sans Bold glyphs (the frame's font)."""
    from PIL import ImageDraw, ImageFont
    import glob
    cands = glob.glob("/usr/share/fonts/**/DejaVuSans-Bold.ttf", recursive=True) + \
        glob.glob("/usr/local/lib/python3*/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf")
    font_path = cands[0] if cands else None
    SZ = 32

    def norm(mask):
        yy, xx = np.where(mask)
        d = mask[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
        im = Image.fromarray((d * 255).astype(np.uint8)).resize((SZ, SZ), Image.BILINEAR)
        return np.array(im).astype(float) / 255.0

    templates = {}
    for digit in range(1, 10):
        im = Image.new("L", (200, 200), 0)
        ImageDraw.Draw(im).text((50, 40), str(digit), fill=255,
                                font=ImageFont.truetype(font_path, 100) if font_path else None)
        templates[digit] = norm(np.array(im) > 128)

    scored = []
    for col, row, dark in wps:
        t = norm(dark)
        best = min(templates, key=lambda k: np.abs(templates[k] - t).mean())
        scored.append((best, col, row, dark))
    scored.sort(key=lambda t: t[0])
    assert [s[0] for s in scored] == list(range(1, len(scored) + 1)), [s[0] for s in scored]
    return [(c, r, d) for _, c, r, d in scored]


if __name__ == "__main__":
    main()
