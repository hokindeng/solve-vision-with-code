#!/usr/bin/env python3
"""LEGO assembly step 4: move the green 1x1 brick from the callout to the
position marked by the red arrow / dashed ghost, and snap it in place.

Everything is drawn procedurally on top of first_frame.png; only the moving
brick, and (at landing time) the red dashed ghost outline, change.
"""
import os
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT = os.path.join(HERE, "output", "video.mp4")

FPS = 16
N_FRAMES = 46

# ---- style measured from first_frame.png (flat colours, no anti-aliasing)
BG = (245, 245, 240)
OUTLINE = (50, 50, 50)
GREEN_TOP = (0, 175, 85)
GREEN_STUD = (0, 192, 93)
GREEN_LEFT = (0, 102, 49)
GREEN_RIGHT = (0, 124, 60)

# ---- brick geometry (isometric 1x1 brick), measured from the ghost outline
HALF_W = 44      # half width of the top rhombus
HALF_H = 25      # half height of the top rhombus
BRICK_H = 61     # vertical extent of the side faces
STUD_RX, STUD_RY = 18.5, 10

# top vertex of the brick in the callout, and at the destination (ghost)
START_TOP = (160.0, 351.0)
DEST_TOP = (781.0, 888.0)
# arrow end points (shaft start inside callout brick, arrowhead tip)
ARROW_START = (160.0, 420.0)
ARROW_TIP = (781.0, 875.0)

HOVER = 45       # how high above the target the brick hovers before dropping
TRAVEL_END = 33  # last frame of the travel phase
DROP_END = 41    # frame at which the brick is snapped in place


def draw_brick(img, top, ghost_alpha=None):
    """Draw a green 1x1 brick whose top vertex is at `top` (x, y)."""
    x, y = int(round(top[0])), int(round(top[1]))
    T = (x, y)
    L = (x - HALF_W, y + HALF_H)
    R = (x + HALF_W, y + HALF_H)
    B = (x, y + 2 * HALF_H)
    L2 = (L[0], L[1] + BRICK_H)
    R2 = (R[0], R[1] + BRICK_H)
    B2 = (B[0], B[1] + BRICK_H)
    d = ImageDraw.Draw(img)
    d.polygon([T, R, B, L], fill=GREEN_TOP)
    d.polygon([L, B, B2, L2], fill=GREEN_LEFT)
    d.polygon([B, R, R2, B2], fill=GREEN_RIGHT)
    # stud on the top face
    cx, cy = x, y + HALF_H
    d.ellipse([cx - STUD_RX, cy - STUD_RY, cx + STUD_RX, cy + STUD_RY],
              fill=GREEN_STUD, outline=OUTLINE, width=1)
    for a, b in [(T, L), (T, R), (L, B), (R, B), (L, L2), (R, R2), (B, B2),
                 (L2, B2), (B2, R2)]:
        d.line([a, b], fill=OUTLINE, width=1)


def arrow_path(first):
    """Return the red arrow's centreline as an arc-length parameterised
    callable P(u) -> (x, y), u in [0, 1], from shaft start to arrowhead tip."""
    a = first.astype(int)
    red = (a[..., 0] == 255) & (a[..., 1] == 0) & (a[..., 2] == 0)
    xs, ys = [], []
    for x in range(170, 715):
        col = np.nonzero(red[:870, x])[0]
        if len(col):
            xs.append(x)
            ys.append(col.mean())
    xs = np.array(xs, float)
    ys = np.array(ys, float)
    # cubic fit, with end points strongly weighted
    X = np.concatenate([xs, [ARROW_START[0]] * 30, [ARROW_TIP[0]] * 30])
    Y = np.concatenate([ys, [ARROW_START[1]] * 30, [ARROW_TIP[1]] * 30])
    coef = np.polyfit(X, Y, 3)
    px = np.linspace(ARROW_START[0], ARROW_TIP[0], 800)
    py = np.polyval(coef, px)
    # pin the ends exactly
    py = py + np.linspace(ARROW_START[1] - py[0], ARROW_TIP[1] - py[-1], len(py))
    seg = np.hypot(np.diff(px), np.diff(py))
    s = np.concatenate([[0], np.cumsum(seg)])
    s /= s[-1]

    def P(u):
        u = float(np.clip(u, 0, 1))
        return float(np.interp(u, s, px)), float(np.interp(u, s, py))

    return P


def make_plate(first):
    """first_frame with the red dashed ghost outline removed."""
    a = first.copy()
    H, W, _ = a.shape
    red = (a[..., 0] == 255) & (a[..., 1] == 0) & (a[..., 2] == 0)
    yy, xx = np.mgrid[0:H, 0:W]
    mask = red & (yy >= 886) & (xx <= 832)
    # fill masked pixels from the nearest unmasked pixel in the same row
    out = a.copy()
    for y in np.unique(np.nonzero(mask)[0]):
        row = a[y]
        m = mask[y]
        good = np.nonzero(~m)[0]
        bad = np.nonzero(m)[0]
        idx = np.searchsorted(good, bad)
        left = good[np.clip(idx - 1, 0, len(good) - 1)]
        right = good[np.clip(idx, 0, len(good) - 1)]
        pick = np.where(np.abs(bad - left) <= np.abs(right - bad), left, right)
        out[y, bad] = row[pick]
    # redraw the yellow brick's outline edges that were hidden under the ghost
    # (endpoints chosen so the line reproduces the visible original pixels
    # exactly; only the erased ghost pixels are touched)
    img = Image.fromarray(out.copy())
    d = ImageDraw.Draw(img)
    d.line([(779, 888), (868, 939)], fill=OUTLINE, width=1)   # top-face left edge
    d.line([(781, 889), (781, 950)], fill=OUTLINE, width=1)   # left vertical edge
    lined = np.array(img)
    out[mask] = lined[mask]
    return out


def smoothstep(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * (3 - 2 * s)


def brick_top_at(frame, P):
    """Top-vertex position of the moving brick for a given frame."""
    if frame <= TRAVEL_END:
        u = smoothstep(frame / TRAVEL_END)
        ax, ay = P(u)
        # offset from the arrow line to the brick's top vertex: at the start it
        # matches the callout brick, at the end the brick hovers above target
        off0 = START_TOP[1] - ARROW_START[1]
        off1 = (DEST_TOP[1] - HOVER) - ARROW_TIP[1]
        return ax + (1 - u) * (START_TOP[0] - ARROW_START[0]) + u * (DEST_TOP[0] - ARROW_TIP[0]), \
            ay + (1 - u) * off0 + u * off1
    if frame <= DROP_END:
        s = (frame - TRAVEL_END) / (DROP_END - TRAVEL_END)
        lift = HOVER * (1 - s * s)            # ease-in (accelerating drop)
        return DEST_TOP[0], DEST_TOP[1] - lift
    return DEST_TOP


def main():
    first = np.array(Image.open(FIRST).convert("RGB"))
    plate = make_plate(first)
    P = arrow_path(first)

    tmpdir = tempfile.mkdtemp(prefix="lego_frames_")
    for f in range(N_FRAMES):
        if f == 0:
            frame = first.copy()
        else:
            # ghost fades out while the brick drops, gone once it is seated
            if f <= TRAVEL_END:
                g = 1.0
            elif f <= DROP_END:
                g = 1.0 - (f - TRAVEL_END) / (DROP_END - TRAVEL_END)
            else:
                g = 0.0
            base = (first.astype(float) * g + plate.astype(float) * (1 - g))
            frame = np.clip(base + 0.5, 0, 255).astype(np.uint8)
            img = Image.fromarray(frame)
            draw_brick(img, brick_top_at(f, P))
            frame = np.array(img)
        Image.fromarray(frame).save(os.path.join(tmpdir, f"{f:03d}.png"))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmpdir, "%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "10", "-preset", "slow",
        "-r", str(FPS), OUT], check=True)
    Image.fromarray(np.array(Image.open(os.path.join(tmpdir, f"{N_FRAMES-1:03d}.png")))).save(
        os.path.join(HERE, "output", "last_frame.png"))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
