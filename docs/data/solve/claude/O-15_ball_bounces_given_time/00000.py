#!/usr/bin/env python3
"""Simulate a ball bouncing twice off the boundary walls, starting from first_frame.png."""
import subprocess, os
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

FPS = 16
N_FRAMES = 79
N_BOUNCES = 2

BALL_COLOR = np.array([72, 61, 139])
ARROW_COLOR = np.array([255, 140, 0])
WALL_COLOR = np.array([100, 100, 100])


def main():
    first = np.array(Image.open(FIRST).convert("RGB"))
    H, W = first.shape[:2]

    ball_mask = np.all(first == BALL_COLOR, axis=2)
    arrow_mask = np.all(first == ARROW_COLOR, axis=2)
    wall_mask = np.all(first == WALL_COLOR, axis=2)

    # Ball geometry (drawn as a PIL ellipse over its bounding box).
    ys, xs = np.nonzero(ball_mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    r = (x1 - x0) / 2.0
    start = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])

    # Wall inner faces: interior spans between the two wall bands.
    wy, wx = np.nonzero(wall_mask)
    mid_row = wall_mask[int(round(start[1]))]
    cols = np.nonzero(mid_row)[0]
    left_face = cols[cols < W // 2].max() + 1        # first interior column
    right_face = cols[cols > W // 2].min()           # first wall column on right
    mid_col = wall_mask[:, W // 2]
    rows = np.nonzero(mid_col)[0]
    top_face = rows[rows < H // 2].max() + 1
    bottom_face = rows[rows > H // 2].min()
    xmin, xmax = left_face + r, right_face - r
    ymin, ymax = top_face + r, bottom_face - r

    # Direction: fit a line through the arrow shaft pixels (near the ball centre),
    # oriented from the ball toward the arrow tip.
    ay, ax = np.nonzero(arrow_mask)
    d = np.hypot(ax - start[0], ay - start[1])
    tip = np.array([ax[d.argmax()], ay[d.argmax()]], float)
    shaft = np.stack([ax, ay], 1)[d < 0.6 * d.max()].astype(float)
    _, _, vt = np.linalg.svd(shaft - shaft.mean(0))
    v = vt[0]
    if np.dot(v, tip - start) < 0:
        v = -v
    v /= np.linalg.norm(v)

    # Simulate straight-line segments with specular reflection.
    pos, vel = start.copy(), v.copy()
    points = [pos.copy()]
    for _ in range(N_BOUNCES):
        ts = []
        if vel[0] > 0: ts.append(((xmax - pos[0]) / vel[0], 0))
        if vel[0] < 0: ts.append(((xmin - pos[0]) / vel[0], 0))
        if vel[1] > 0: ts.append(((ymax - pos[1]) / vel[1], 1))
        if vel[1] < 0: ts.append(((ymin - pos[1]) / vel[1], 1))
        t, axis = min(ts, key=lambda p: p[0])
        pos = pos + vel * t
        # Handle simultaneous corner hit: flip every axis that reached its limit.
        for a in (0, 1):
            lim = (xmin, xmax) if a == 0 else (ymin, ymax)
            if abs(pos[a] - lim[0]) < 1e-6 or abs(pos[a] - lim[1]) < 1e-6:
                vel[a] = -vel[a]
        points.append(pos.copy())

    seg_len = [np.linalg.norm(points[i + 1] - points[i]) for i in range(len(points) - 1)]
    total = sum(seg_len)

    def pos_at(s):
        for i, L in enumerate(seg_len):
            if s <= L or i == len(seg_len) - 1:
                f = min(s / L, 1.0) if L > 0 else 1.0
                return points[i] + (points[i + 1] - points[i]) * f
            s -= L
        return points[-1]

    # Background: first frame with the ball removed (background is white),
    # arrow pixels re-drawn on top of the ball, as in the original frame.
    bg = first.copy()
    bg[ball_mask] = 255
    arrow_pixels = first[arrow_mask]

    os.makedirs(OUT_DIR, exist_ok=True)
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
         "-r", str(FPS), OUT],
        stdin=subprocess.PIPE)

    for i in range(N_FRAMES):
        s = total * i / (N_FRAMES - 1)
        c = pos_at(s)
        cx, cy = int(round(c[0])), int(round(c[1]))
        img = Image.fromarray(bg.copy())
        ImageDraw.Draw(img).ellipse(
            (cx - int(r), cy - int(r), cx + int(r), cy + int(r)), fill=tuple(BALL_COLOR))
        frame = np.array(img)
        frame[arrow_mask] = arrow_pixels
        ffmpeg.stdin.write(frame.tobytes())
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print("start", start, "dir", v, "bounces", points[1:], "written", OUT)


if __name__ == "__main__":
    main()
