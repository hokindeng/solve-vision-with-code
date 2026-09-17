#!/usr/bin/env python3
"""Rotate the blue polygon 42 degrees counterclockwise about the marked center
until it coincides with the dashed target outline. Everything else in the
frame is left untouched (pixels are copied from first_frame.png)."""
import subprocess
import numpy as np
from PIL import Image, ImageDraw

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
W = H = 1024
FPS = 16
N_FRAMES = 70

BG = (240, 240, 240)
FILL = (77, 147, 189)
EDGE = (50, 50, 50)
DASH = (100, 100, 100)

CENTER = np.array([572.0, 589.0])
# Polygon vertices (the third one sits exactly on the rotation center).
VERTS = np.array([[635, 478], [560, 545], [572, 589], [658, 565]], float)
# Counterclockwise on screen == negative angle in image (y-down) coordinates.
TOTAL_ANGLE = np.radians(-42.0)

DASH_LEN, GAP_LEN, DASH_W = 12.0, 4.0, 4


def rotate(pts, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return (pts - CENTER) @ R.T + CENTER


def draw_dashed_polygon(draw, pts):
    """Dashed closed outline with the dash phase running continuously around
    the perimeter (order matches the visible dashes in the first frame)."""
    order = [0, 3, 2, 1]
    phase = 0.0
    for k in range(4):
        a = pts[order[k]]
        b = pts[order[(k + 1) % 4]]
        d = b - a
        L = np.linalg.norm(d)
        u = d / L
        s = 0.0
        while s < L:
            period_pos = phase % (DASH_LEN + GAP_LEN)
            if period_pos < DASH_LEN:
                seg = min(DASH_LEN - period_pos, L - s)
                p0 = a + u * s
                p1 = a + u * (s + seg)
                draw.line([tuple(p0), tuple(p1)], fill=DASH, width=DASH_W)
            else:
                seg = min(DASH_LEN + GAP_LEN - period_pos, L - s)
            s += seg
            phase += seg


def ease(t):
    return t * t * (3 - 2 * t)  # smoothstep


def main():
    first = np.array(Image.open(SRC).convert("RGB"))

    poly_mask = np.all(first == FILL, axis=2) | np.all(first == EDGE, axis=2)
    marker_mask = np.all(first == (0, 0, 0), axis=2) | np.all(first == (255, 255, 255), axis=2)

    # Static background: original frame with the polygon erased. The part of
    # the dashed target hidden underneath the polygon is synthesized.
    base = first.copy()
    base[poly_mask] = BG
    target = rotate(VERTS, TOTAL_ANGLE)
    dash_img = Image.new("RGB", (W, H), BG)
    draw_dashed_polygon(ImageDraw.Draw(dash_img), target)
    dash_arr = np.array(dash_img)
    hidden = poly_mask & np.all(dash_arr == DASH, axis=2)
    base[hidden] = DASH

    frames = []
    for i in range(N_FRAMES):
        if i == 0:
            frames.append(first)
            continue
        t = ease(i / (N_FRAMES - 1))
        pts = rotate(VERTS, TOTAL_ANGLE * t)
        img = Image.fromarray(base.copy())
        d = ImageDraw.Draw(img)
        d.polygon([tuple(p) for p in pts], fill=FILL, outline=EDGE)
        arr = np.array(img)
        arr[marker_mask] = first[marker_mask]  # marker stays on top
        frames.append(arr)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-movflags", "+faststart", OUT,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    # Also save the last frame for inspection.
    Image.fromarray(frames[-1]).save("/app/output/last_frame.png")


if __name__ == "__main__":
    main()
