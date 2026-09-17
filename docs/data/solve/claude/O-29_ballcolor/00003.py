#!/usr/bin/env python3
"""Regenerate /app/output/video.mp4.

Scene: red cluster A (11), green cluster C (10), violet cluster B (4), TOTAL = 25.
Rule: A can only absorb clusters smaller than itself.
Solution: A(11) absorbs B(4) -> A(15); A(15) absorbs C(10) -> A(25).

The scene is re-rendered from scratch each frame using geometry measured from
first_frame.png (frame 0 reproduces it pixel-exactly). Only the absorbed balls,
their labels and A's count label change; everything else stays untouched.
"""
import os
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

APP = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

W = H = 1024
FPS = 16
N_FRAMES = 61

BG = (240, 240, 240)
SHADOW = (50, 50, 50)
RED = (255, 50, 50)
GREEN = (152, 251, 152)
VIOLET = (238, 130, 238)
R = 20
STEP = 50

# Cluster geometry measured from first_frame.png
A_ORIGIN = (323, 463)           # top-left ball centre, 3 columns
C_ORIGIN = (189, 199)           # 3 columns
B_ORIGIN = (331, 818)           # 2 columns
A_COLS, C_COLS, B_COLS = 3, 3, 2
A_N, C_N, B_N = 11, 10, 4

# Labels: (text origin, box padding)
TOTAL_LABEL = ("TOTAL = 25", (10, 10), 5)
C_LABEL = ("C: 10", (355, 264), 3)
A_LABEL_ORIGIN = (489, 528)
B_LABEL = ("B: 4", (417, 833), 3)

font = ImageFont.truetype(FONT, 48)


def grid_pos(origin, cols, i):
    return (origin[0] + STEP * (i % cols), origin[1] + STEP * (i // cols))


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)  # smoothstep


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_color(c0, c1, t):
    return tuple(int(round(lerp(a, b, t))) for a, b in zip(c0, c1))


def bezier(p0, p1, p2, t):
    x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0]
    y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]
    return x, y


def draw_ball(d, cx, cy, color):
    cx, cy = int(round(cx)), int(round(cy))
    d.ellipse([cx - R + 1, cy - R + 1, cx + R + 1, cy + R + 1], fill=SHADOW)
    d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=color)


def draw_label(img, text, origin, pad, alpha=1.0):
    """Draw a white box + black text. alpha<1 fades the label into the background."""
    if alpha <= 0:
        return
    layer = Image.new("RGB", img.size, BG)
    d = ImageDraw.Draw(layer)
    bb = d.textbbox(origin, text, font=font)
    box = [bb[0] - pad, bb[1] - pad, bb[2] + pad, bb[3] + pad]
    d.rectangle(box, fill=(255, 255, 255))
    d.text(origin, text, font=font, fill=(0, 0, 0))
    crop = (box[0], box[1], box[2] + 1, box[3] + 1)   # PIL crop is exclusive
    if alpha >= 1:
        img.paste(layer.crop(crop), (box[0], box[1]))
    else:
        blended = Image.blend(img.crop(crop), layer.crop(crop), alpha)
        img.paste(blended, (box[0], box[1]))


# ---------------------------------------------------------------- schedule
# Phase 1: B (4 balls) flies into A slots 11..14.  Phase 2: C (10 balls) -> slots 15..24.
P1_START, P1_TRAVEL, P1_STAGGER = 5, 15, 2      # last arrival: 5 + 3*2 + 15 = 26
P2_START, P2_TRAVEL, P2_STAGGER = 31, 17, 1     # last arrival: 31 + 9*1 + 17 = 57
FADE = 6                                         # label fade-out length (frames)

P1_END = P1_START + (B_N - 1) * P1_STAGGER + P1_TRAVEL
P2_END = P2_START + (C_N - 1) * P2_STAGGER + P2_TRAVEL
assert P2_END <= N_FRAMES - 2


def moving_balls(frame, origin, cols, n, color, start, travel, stagger, slot0, ctrl_offset):
    """Yield (x, y, color, progress) for each ball of an absorbed cluster."""
    # farthest-from-A balls leave first so the cluster peels away smoothly
    order = sorted(range(n), key=lambda i: -abs(grid_pos(origin, cols, i)[1] - A_ORIGIN[1]))
    for k, i in enumerate(order):
        src = grid_pos(origin, cols, i)
        dst = grid_pos(A_ORIGIN, A_COLS, slot0 + k)
        t = ease((frame - (start + k * stagger)) / travel)
        mid = ((src[0] + dst[0]) / 2 + ctrl_offset[0], (src[1] + dst[1]) / 2 + ctrl_offset[1])
        x, y = bezier(src, mid, dst, t)
        ct = ease((t - 0.35) / 0.5)          # colour turns red during the flight
        yield x, y, lerp_color(color, RED, ct), t


def render(frame):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # --- static red cluster A
    for i in range(A_N):
        draw_ball(d, *grid_pos(A_ORIGIN, A_COLS, i), RED)

    # --- cluster B (violet) -> absorbed in phase 1
    b_balls = list(moving_balls(frame, B_ORIGIN, B_COLS, B_N, VIOLET,
                                P1_START, P1_TRAVEL, P1_STAGGER, A_N, (0, 0)))
    # --- cluster C (green) -> absorbed in phase 2; path swings left around A
    c_balls = list(moving_balls(frame, C_ORIGIN, C_COLS, C_N, GREEN,
                                P2_START, P2_TRAVEL, P2_STAGGER, A_N + B_N, (-260, 0)))

    # draw balls that have landed first, then the in-flight ones on top
    for group in (b_balls, c_balls):
        for x, y, col, t in group:
            if t >= 1.0:
                draw_ball(d, x, y, col)
    for group in (b_balls, c_balls):
        for x, y, col, t in group:
            if t < 1.0:
                draw_ball(d, x, y, col)

    # --- labels
    draw_label(img, *TOTAL_LABEL)
    draw_label(img, C_LABEL[0], C_LABEL[1], C_LABEL[2],
               alpha=1.0 - ease((frame - P2_START) / FADE))
    draw_label(img, B_LABEL[0], B_LABEL[1], B_LABEL[2],
               alpha=1.0 - ease((frame - P1_START) / FADE))
    a_count = A_N + (B_N if frame >= P1_END else 0) + (C_N if frame >= P2_END else 0)
    draw_label(img, f"A: {a_count}", A_LABEL_ORIGIN, 3)
    return img


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = [render(f) for f in range(N_FRAMES)]

    ref_path = os.path.join(APP, "first_frame.png")
    if os.path.exists(ref_path):
        ref = np.array(Image.open(ref_path).convert("RGB")).astype(int)
        diff = int((np.abs(np.array(frames[0]).astype(int) - ref).sum(axis=2) > 0).sum())
        print(f"frame 0 vs first_frame.png: {diff} differing pixels")

    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "10", "-pix_fmt", "yuv420p",
           "-r", str(FPS), OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        proc.stdin.write(fr.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    frames[-1].save(os.path.join(OUT_DIR, "last_frame.png"))
    print(f"wrote {OUT} ({N_FRAMES} frames @ {FPS} fps)")


if __name__ == "__main__":
    main()
