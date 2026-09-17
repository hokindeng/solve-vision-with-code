#!/usr/bin/env python3
"""Generate the A:B :: C:? shape-style analogy video.

Row 1: solid-outline circle -> dotted (thin, dashed) outline circle.
Row 2: solid-outline arrow  -> ?   ==> the arrow is drawn in the right cell
       and its outline is converted, sweeping around the perimeter, from a
       solid 8px stroke to a dotted 4px stroke (dash 6 / gap 2, like row 1).
Everything outside the right cell of row 2 is left untouched.
"""
import os
import subprocess
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

W = H = 1024
FPS = 16
N_FRAMES = 60

BLUE = np.array([9, 100, 191], dtype=np.float32)
WHITE = np.array([255, 255, 255], dtype=np.float32)

# geometry of the arrow in the left cell (measured from first_frame.png)
DX = 704  # shift from left cell (x centre 160) to right cell (x centre 864)
ARROW = [  # centreline polygon (closed), left-cell coordinates
    (79.5, 728.5), (159.5, 728.5), (159.5, 688.0), (240.0, 768.0),
    (159.5, 848.0), (159.5, 807.5), (79.5, 807.5),
]
SOLID_HW = 4.0   # half stroke width of solid style (8 px)
DOT_HW = 2.0     # half stroke width of dotted style (4 px)
DASH_ON = 6.0
DASH_PERIOD = 8.0

# region of the right cell that we are allowed to touch
X0, X1, Y0, Y1 = 740, 1000, 650, 890
# bounding box of the "?" glyph
QX0, QX1, QY0, QY1 = 851, 878, 746, 790


def smoothstep(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def path_fields(pts, xs, ys):
    """For every pixel centre return (distance to path, arc-length of nearest point)."""
    best_d = np.full(xs.shape, np.inf, dtype=np.float64)
    best_s = np.zeros(xs.shape, dtype=np.float64)
    acc = 0.0
    n = len(pts)
    for i in range(n):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % n]
        vx, vy = bx - ax, by - ay
        seg_len = float(np.hypot(vx, vy))
        t = ((xs - ax) * vx + (ys - ay) * vy) / (seg_len ** 2)
        t = np.clip(t, 0.0, 1.0)
        px, py = ax + t * vx, ay + t * vy
        d = np.hypot(xs - px, ys - py)
        better = d < best_d
        best_d = np.where(better, d, best_d)
        best_s = np.where(better, acc + t * seg_len, best_s)
        acc += seg_len
    return best_d, best_s, acc


def main():
    base = np.array(Image.open(FIRST).convert("RGB")).astype(np.float32)

    # --- prepare masks for the right cell -----------------------------------
    ys, xs = np.mgrid[Y0:Y1, X0:X1]
    xs = xs.astype(np.float64) + 0.5
    ys = ys.astype(np.float64) + 0.5
    pts = [(x + DX, y) for x, y in ARROW]
    dist, arc, total = path_fields(pts, xs, ys)

    # solid arrow: exact pixel copy of the left-cell arrow, translated right
    left = base[Y0:Y1, X0 - DX:X1 - DX]
    solid_mask = np.abs(left - WHITE).sum(axis=2) > 60

    # dotted arrow: 4 px stroke, dash 6 / gap 2 (integer number of periods)
    n_per = max(1, round(total / DASH_PERIOD))
    period = total / n_per
    on = DASH_ON / DASH_PERIOD * period
    dotted_mask = (dist <= DOT_HW) & ((arc % period) < on)

    # the "?" glyph pixels
    q_region = base[Y0:Y1, X0:X1]
    q_mask = np.zeros(solid_mask.shape, dtype=bool)
    q_mask[QY0 - Y0:QY1 - Y0, QX0 - X0:QX1 - X0] = (
        np.abs(q_region[QY0 - Y0:QY1 - Y0, QX0 - X0:QX1 - X0] - WHITE).sum(axis=2) > 0
    )
    q_colors = q_region.copy()

    # --- timeline -----------------------------------------------------------
    Q_FADE = (1, 12)      # "?" fades out
    A_FADE = (8, 21)      # solid arrow fades in
    SWEEP = (22, 58)      # outline converted solid -> dotted around perimeter

    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for f in range(N_FRAMES):
        img = base.copy()
        cell = np.tile(WHITE, (Y1 - Y0, X1 - X0, 1))

        # "?" (fading out)
        qa = 1.0 - smoothstep((f - Q_FADE[0]) / (Q_FADE[1] - Q_FADE[0]))
        if f < Q_FADE[0]:
            qa = 1.0
        if qa > 0:
            cell[q_mask] = WHITE + (q_colors[q_mask] - WHITE) * qa

        # arrow (fading in, then converting)
        aa = smoothstep((f - A_FADE[0]) / (A_FADE[1] - A_FADE[0]))
        if f < A_FADE[0]:
            aa = 0.0
        if aa > 0:
            if f < SWEEP[0]:
                sweep = 0.0
            elif f >= SWEEP[1]:
                sweep = total + 1.0
            else:
                sweep = total * smoothstep((f - SWEEP[0]) / (SWEEP[1] - SWEEP[0]))
            converted = arc <= sweep
            mask = (dotted_mask & converted) | (solid_mask & ~converted)
            col = WHITE + (BLUE - WHITE) * aa
            cell[mask] = col

        img[Y0:Y1, X0:X1] = cell
        frames.append(np.clip(img + 0.5, 0, 255).astype(np.uint8))

    # frame 0 must be the untouched first frame
    frames[0] = base.astype(np.uint8)

    # --- encode -------------------------------------------------------------
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-tune", "stillimage", "-g", "16", OUT,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        proc.stdin.write(fr.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {OUT} ({len(frames)} frames @ {FPS} fps)")


if __name__ == "__main__":
    main()
