#!/usr/bin/env python3
"""Regenerate /app/output/video.mp4: red cluster A absorbs B (2), then C (6), then D (12).

Absorption order must always target a cluster smaller than A:
  A=5  -> absorb B(2)  -> 7
  A=7  -> absorb C(6)  -> 13
  A=13 -> absorb D(12) -> 25
"""
import math
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

W = H = 1024
FPS = 16
N_FRAMES = 80

BG = (240, 240, 240)
SHADOW = (50, 50, 50)
R = 18            # ball radius (diameter 37 px inclusive)
SH_OFF = 1        # shadow offset
SPACING = 45
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
LABEL_PAD = 3
LABEL_DX = 33     # text origin x = cluster right edge + 33
LABEL_DY = -10    # text origin y = cluster vertical centre - 10

# ---- scene as measured from first_frame.png -------------------------------
CLUSTERS = {
    "A": dict(color=(255, 50, 50), cols=2, origin=(292, 233), n=5),
    "B": dict(color=(255, 220, 50), cols=1, origin=(411, 727), n=2),
    "C": dict(color=(64, 224, 208), cols=2, origin=(510, 448), n=6),
    "D": dict(color=(200, 50, 255), cols=3, origin=(93, 430), n=12),
}
# measured text origins of the labels in the first frame
LABEL_ORIGIN = {"A": (388, 268), "B": (456, 740), "C": (606, 483), "D": (242, 488)}

ORDER = ["B", "C", "D"]   # the only valid absorption order


def grid(origin, cols, n, spacing=SPACING):
    ox, oy = origin
    return [(ox + (i % cols) * spacing, oy + (i // cols) * spacing) for i in range(n)]


def merged_layout(n):
    """Layout of the red cluster once it holds n balls: anchored at A's top-left ball."""
    cols = CLUSTERS["A"]["cols"] if n == CLUSTERS["A"]["n"] else math.ceil(math.sqrt(n))
    return grid(CLUSTERS["A"]["origin"], cols, n)


def label_origin_for(positions):
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    right = max(xs) + R
    cy = (min(ys) + max(ys)) / 2.0
    return (right + LABEL_DX, int(round(cy + LABEL_DY)))


def smoothstep(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_col(c0, c1, t):
    return tuple(int(round(lerp(a, b, t))) for a, b in zip(c0, c1))


def draw_ball(d, x, y, color):
    x, y = int(round(x)), int(round(y))
    d.ellipse([x - R + SH_OFF, y - R + SH_OFF, x + R + SH_OFF, y + R + SH_OFF], fill=SHADOW)
    d.ellipse([x - R, y - R, x + R, y + R], fill=color)


def draw_label(d, origin, text):
    x0, y0, x1, y1 = d.textbbox(origin, text, font=FONT)
    d.rectangle([x0 - LABEL_PAD, y0 - LABEL_PAD, x1 + LABEL_PAD, y1 + LABEL_PAD], fill=(255, 255, 255))
    d.text(origin, text, font=FONT, fill=(0, 0, 0))


def make_background(first):
    """First frame with the four clusters and their labels blanked out (TOTAL label untouched)."""
    bg = first.copy()
    d = ImageDraw.Draw(bg)
    for k, c in CLUSTERS.items():
        for (x, y) in grid(c["origin"], c["cols"], c["n"]):
            d.rectangle([x - R, y - R, x + R + SH_OFF, y + R + SH_OFF], fill=BG)
        x0, y0, x1, y1 = d.textbbox(LABEL_ORIGIN[k], f"{k}: {c['n']}", font=FONT)
        d.rectangle([x0 - LABEL_PAD, y0 - LABEL_PAD, x1 + LABEL_PAD, y1 + LABEL_PAD], fill=BG)
    return bg


# ---- timeline --------------------------------------------------------------
# frame 0: initial.  Then three steps of equal length; each step = flight + hold.
STEP_LEN = (N_FRAMES - 1) // len(ORDER)         # 26 frames per step
FLIGHT = 18                                     # frames of motion per step


def scene_at(frame):
    """Return (balls, labels): balls = list of (x, y, color); labels = list of (origin, text)."""
    red = CLUSTERS["A"]["color"]
    # red cluster state before this frame's step
    red_n = CLUSTERS["A"]["n"]
    red_pos = merged_layout(red_n)
    red_label = LABEL_ORIGIN["A"]
    remaining = list(ORDER)

    balls, labels = [], []
    step_idx = min((frame - 1) // STEP_LEN, len(ORDER) - 1) if frame > 0 else -1

    # fully completed steps
    for s in range(step_idx):
        k = ORDER[s]
        red_n += CLUSTERS[k]["n"]
        red_pos = merged_layout(red_n)
        red_label = label_origin_for(red_pos)
        remaining.remove(k)

    if step_idx >= 0:
        k = ORDER[step_idx]
        remaining.remove(k)  # its label vanishes as soon as the step starts
        src = CLUSTERS[k]
        local = frame - 1 - step_idx * STEP_LEN
        t = smoothstep(local / (FLIGHT - 1)) if local < FLIGHT else 1.0
        new_n = red_n + src["n"]
        new_pos = merged_layout(new_n)
        new_label = label_origin_for(new_pos)
        src_pos = grid(src["origin"], src["cols"], src["n"])
        # existing red balls slide to their new slots
        for i, (x0, y0) in enumerate(red_pos):
            x1, y1 = new_pos[i]
            balls.append((lerp(x0, x1, t), lerp(y0, y1, t), red))
        # absorbed balls fly in and turn red
        for j, (x0, y0) in enumerate(src_pos):
            x1, y1 = new_pos[red_n + j]
            ct = smoothstep((t - 0.55) / 0.45)  # recolour during the last part of the flight
            balls.append((lerp(x0, x1, t), lerp(y0, y1, t), lerp_col(src["color"], red, ct)))
        lab_origin = (int(round(lerp(red_label[0], new_label[0], t))),
                      int(round(lerp(red_label[1], new_label[1], t))))
        labels.append((lab_origin, f"A: {new_n if t >= 1.0 else red_n}"))
    else:
        for (x, y) in red_pos:
            balls.append((x, y, red))
        labels.append((red_label, f"A: {red_n}"))

    # untouched clusters
    for k in remaining:
        c = CLUSTERS[k]
        for (x, y) in grid(c["origin"], c["cols"], c["n"]):
            balls.append((x, y, c["color"]))
        labels.append((LABEL_ORIGIN[k], f"{k}: {c['n']}"))
    return balls, labels


def render(frame, background):
    im = background.copy()
    d = ImageDraw.Draw(im)
    balls, labels = scene_at(frame)
    for (x, y, col) in balls:
        draw_ball(d, x, y, col)
    for origin, text in labels:
        draw_label(d, origin, text)
    return im


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    first = Image.open(FIRST).convert("RGB")
    background = make_background(first)

    frames = [render(f, background) for f in range(N_FRAMES)]
    # frame 0 must reproduce the given first frame exactly
    diff = np.abs(np.asarray(frames[0]).astype(int) - np.asarray(first).astype(int))
    assert diff.max() == 0, f"frame 0 mismatch: {int((diff.sum(2) > 0).sum())} pixels differ"

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-g", "8", "-r", str(FPS), OUT,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(np.asarray(fr, dtype=np.uint8).tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {OUT} ({N_FRAMES} frames @ {FPS} fps)")


if __name__ == "__main__":
    main()
