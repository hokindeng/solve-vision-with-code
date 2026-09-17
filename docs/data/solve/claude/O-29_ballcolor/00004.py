#!/usr/bin/env python3
"""Regenerate /app/output/video.mp4: red cluster A absorbs clusters B, C, D in the
only legal order (each absorbed cluster must be smaller than A at that moment).

    A=5 absorbs B=2  -> A=7
    A=7 absorbs C=6  -> A=13
    A=13 absorbs D=12 -> A=25   (TOTAL = 25, all balls red)

Every frame starts from first_frame.png; only the pixels of cluster A and of the
cluster currently being absorbed are redrawn.  All other pixels stay untouched.
"""
import math
import os
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT = os.path.join(APP, "output", "video.mp4")

FPS = 16
N_FRAMES = 75
W = H = 1024
BG = (240, 240, 240)
RADIUS = 21
SHADOW = (50, 50, 50)
RED = (255, 50, 50)
SPACING = 52.5
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
PAD = 3

# Scene as measured from first_frame.png (integer ball centres, PIL ellipse
# bbox = [cx-21, cy-21, cx+21, cy+21], shadow drawn first at +1,+1 offset).
CLUSTERS = {
    "A": dict(color=RED, label_origin=(563, 498), label="A: 5",
              balls=[(452, 456), (504, 456), (452, 508), (504, 508), (452, 561)]),
    "B": dict(color=(50, 180, 180), label_origin=(440, 223), label="B: 2",
              balls=[(387, 206), (387, 259)]),
    "C": dict(color=(75, 0, 130), label_origin=(291, 532), label="C: 6",
              balls=[(180, 489), (233, 489), (180, 542), (233, 542), (180, 594), (233, 594)]),
    "D": dict(color=(0, 0, 128), label_origin=(640, 824), label="D: 12",
              balls=[(x, y) for y in (755, 808, 860, 913) for x in (466, 519, 571)]),
}
ABSORB_ORDER = ["B", "C", "D"]  # 2 < 5, 6 < 7, 12 < 13
A_ANCHOR = (452, 456)  # top-left ball of cluster A stays the grid anchor


def grid_slots(n):
    """Slot centres for n balls in A: floor(sqrt(n)) columns, filled row by row."""
    cols = max(1, int(math.floor(math.sqrt(n))))
    slots = []
    for i in range(n):
        r, c = divmod(i, cols)
        # floor keeps the 5 original A balls exactly where they are in first_frame.png
        slots.append((math.floor(A_ANCHOR[0] + SPACING * c), math.floor(A_ANCHOR[1] + SPACING * r)))
    return slots


def a_label_origin(slots):
    xs = [s[0] for s in slots]
    ys = [s[1] for s in slots]
    cx_max = int(round(max(xs)))
    cy_mid = (min(ys) + max(ys)) / 2.0
    return (cx_max + RADIUS + 38, int(round(cy_mid)) - 10)


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * t)


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_col(c0, c1, t):
    return tuple(int(round(lerp(a, b, t))) for a, b in zip(c0, c1))


def draw_ball(d, cx, cy, color):
    cx, cy = int(round(cx)), int(round(cy))
    d.ellipse([cx - RADIUS + 1, cy - RADIUS + 1, cx + RADIUS + 1, cy + RADIUS + 1], fill=SHADOW)
    d.ellipse([cx - RADIUS, cy - RADIUS, cx + RADIUS, cy + RADIUS], fill=color)


def draw_label(d, origin, text):
    bb = d.textbbox(origin, text, font=FONT)
    d.rectangle([bb[0] - PAD, bb[1] - PAD, bb[2] + PAD, bb[3] + PAD], fill=(255, 255, 255))
    d.text(origin, text, font=FONT, fill=(0, 0, 0))


def erase_cluster(d, name):
    """Paint background over a cluster's original balls and label (bg is uniform)."""
    cl = CLUSTERS[name]
    for cx, cy in cl["balls"]:
        d.rectangle([cx - RADIUS - 1, cy - RADIUS - 1, cx + RADIUS + 2, cy + RADIUS + 2], fill=BG)
    bb = d.textbbox(cl["label_origin"], cl["label"], font=FONT)
    d.rectangle([bb[0] - PAD - 1, bb[1] - PAD - 1, bb[2] + PAD + 1, bb[3] + PAD + 1], fill=BG)


def verify_static_reproduction(base):
    """Sanity check: redrawing the scene with our primitives reproduces the input."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    for name in CLUSTERS:
        erase_cluster(d, name)
    for name, cl in CLUSTERS.items():
        for cx, cy in cl["balls"]:
            draw_ball(d, cx, cy, cl["color"])
        draw_label(d, cl["label_origin"], cl["label"])
    a = np.array(img).astype(int)
    b = np.array(base).astype(int)
    # ignore the TOTAL label region, which we never redraw
    a[0:70, 0:340] = b[0:70, 0:340]
    diff = int((np.abs(a - b).sum(2) > 0).sum())
    print(f"static reproduction mismatch pixels: {diff}")


def render_frame(base, f):
    img = base.copy()
    if f == 0:
        return img
    d = ImageDraw.Draw(img)

    # Timeline: frames 1..72 hold three 24-frame absorption steps, 73..74 hold the result.
    step_len = 24
    step = min((f - 1) // step_len, len(ABSORB_ORDER) - 1)
    t = ease(((f - 1) % step_len) / (step_len - 1)) if f <= step_len * len(ABSORB_ORDER) else 1.0

    # Balls currently belonging to A (before this step) with their colour.
    a_balls = list(CLUSTERS["A"]["balls"])
    for s in range(step):
        a_balls += CLUSTERS[ABSORB_ORDER[s]]["balls"]
    n_before = len(a_balls)
    target = ABSORB_ORDER[step]
    src = CLUSTERS[target]
    n_after = n_before + len(src["balls"])

    slots_before = grid_slots(n_before)
    slots_after = grid_slots(n_after)

    # Erase everything that is being redrawn: A, already absorbed clusters, current target.
    erase_cluster(d, "A")
    for s in range(step + 1):
        erase_cluster(d, ABSORB_ORDER[s])

    # Existing red balls glide from their old slot to their new slot.
    for i in range(n_before):
        x = lerp(slots_before[i][0], slots_after[i][0], t)
        y = lerp(slots_before[i][1], slots_after[i][1], t)
        draw_ball(d, x, y, RED)
    # Absorbed balls fly in and turn red on the way.
    for j, (sx, sy) in enumerate(src["balls"]):
        dx, dy = slots_after[n_before + j]
        x, y = lerp(sx, dx, t), lerp(sy, dy, t)
        draw_ball(d, x, y, lerp_col(src["color"], RED, t))

    # Label: count updates once the balls have arrived; the box follows the grid.
    if t >= 1.0:
        draw_label(d, a_label_origin(slots_after), f"A: {n_after}")
    else:
        draw_label(d, a_label_origin(slots_before), f"A: {n_before}")
    return img


def main():
    base = Image.open(FIRST).convert("RGB")
    assert base.size == (W, H)
    verify_static_reproduction(base)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="frames_")
    try:
        for f in range(N_FRAMES):
            render_frame(base, f).save(os.path.join(tmp, f"{f:03d}.png"))
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(FPS), "-i", os.path.join(tmp, "%03d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "10",
            "-preset", "slow", "-r", str(FPS), OUT,
        ]
        subprocess.run(cmd, check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
