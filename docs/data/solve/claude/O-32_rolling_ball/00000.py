#!/usr/bin/env python3
"""Animate the yellow ball rolling along the dashed platform path.

The first frame is used verbatim as the background.  The ball (a PIL ellipse
with a 2px darker outline) is lifted off the background, the platform it hides
is restored, and the ball is then redrawn at positions that ease along the
straight line through the platform centres, finishing centred on the last one.
"""
import os
import subprocess
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

W = H = 1024
FPS = 16
N_FRAMES = 64

BALL_FILL = (255, 255, 0)
BALL_EDGE = (175, 175, 0)
BALL_R = 40                     # bbox [cx-40, cy-40, cx+40, cy+40]
BALL_START = (368.0, 592.0)     # centre in first frame (bbox 328..408, 552..632)

DASH_FILL = (70, 130, 200)
DASH_EDGE = (10, 70, 140)


def load_first_frame():
    return np.array(Image.open(SRC).convert("RGB"))


def platform_centres(img):
    """Centroids of the blue dashes, ordered from the ball outwards."""
    import cv2
    blue = ((img[..., 2] > 150) & (img[..., 0] < 120)).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(blue)
    pts = [tuple(cent[i]) for i in range(1, n)]
    bx, by = BALL_START
    pts.sort(key=lambda p: (p[0] - bx) ** 2 + (p[1] - by) ** 2)
    return pts


def build_background(img):
    """Remove the ball and restore the platform it partially covers."""
    r, g, b = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
    ball_mask = ((r == 255) & (g == 255) & (b == 0)) | ((r == 175) & (g == 175) & (b == 0))
    bg = img.copy()
    bg[ball_mask] = 255

    # Restore the hidden part of the first dash by copying the second dash,
    # shifted by the inter-dash step, into the pixels the ball used to cover.
    pts = platform_centres(img)
    p1 = np.array(pts[1]); p2 = np.array(pts[2])
    step = p1 - p2                                  # vector from dash 2 to dash 1
    import cv2
    dash_mask = ((b > 150) & (r < 120)).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(dash_mask)
    lab2 = min(range(1, n), key=lambda k: (cent[k][0] - p1[0]) ** 2 + (cent[k][1] - p1[1]) ** 2)
    ys2, xs2 = np.nonzero(lab == lab2)           # pixels of the second dash only
    best = None
    for dx in (int(np.floor(step[0])), int(np.ceil(step[0]))):
        for dy in (int(np.floor(step[1])), int(np.ceil(step[1]))):
            tx, ty = xs2 + dx, ys2 + dy
            ok = (tx >= 0) & (tx < W) & (ty >= 0) & (ty < H)
            vis = ~ball_mask[ty[ok], tx[ok]]
            score = np.mean(np.all(img[ty[ok][vis], tx[ok][vis]] == img[ys2[ok][vis], xs2[ok][vis]], axis=1))
            if best is None or score > best[0]:
                best = (score, dx, dy)
    _, dx, dy = best
    tx, ty = xs2 + dx, ys2 + dy
    hidden = ball_mask[ty, tx]
    bg[ty[hidden], tx[hidden]] = img[ys2[hidden], xs2[hidden]]
    return bg, pts


def ease(t):
    """Smooth ease-in-out (smootherstep)."""
    t = min(max(t, 0.0), 1.0)
    return t * t * t * (t * (6 * t - 15) + 10)


def ball_centre(i, start, end):
    """Ball centre on frame i: roll from start to end, then rest."""
    travel_frames = 58          # arrive here, rest for the remaining frames
    t = ease(i / (travel_frames - 1))
    x = start[0] + (end[0] - start[0]) * t
    y = start[1] + (end[1] - start[1]) * t
    return x, y


def draw_ball(bg, cx, cy):
    im = Image.fromarray(bg)
    d = ImageDraw.Draw(im)
    x0 = int(round(cx)) - BALL_R
    y0 = int(round(cy)) - BALL_R
    d.ellipse([x0, y0, x0 + 2 * BALL_R, y0 + 2 * BALL_R],
              fill=BALL_FILL, outline=BALL_EDGE, width=2)
    return np.array(im)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    first = load_first_frame()
    bg, pts = build_background(first)
    end = pts[-1]                       # rest centred on the final platform

    frames = []
    for i in range(N_FRAMES):
        if i == 0:
            frames.append(first)
            continue
        cx, cy = ball_centre(i, BALL_START, end)
        frames.append(draw_ball(bg, cx, cy))

    # sanity: redrawing frame 0 reproduces the source exactly
    assert np.array_equal(draw_ball(bg, *BALL_START), first), "frame-0 mismatch"

    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
           "-movflags", "+faststart", OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {OUT} ({N_FRAMES} frames @ {FPS} fps)")


if __name__ == "__main__":
    main()
