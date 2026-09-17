#!/usr/bin/env python3
"""Ball-cluster absorption video.

Red cluster A (3 balls) absorbs the other clusters in the only legal order
(smaller than A each time): B(2) -> A=5, C(4) -> A=9, D(5) -> A=14.
Everything else in the frame (background, TOTAL label) is left untouched.
"""
import math
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

W = H = 1024
FPS = 16
N_FRAMES = 80
BG = (240, 240, 240)
R = 21                      # ball radius (PIL ellipse bbox cx-21..cx+21)
SHADOW = (50, 50, 50)       # shadow ellipse, offset (+1,+1)
SPACING = 53
RED = (255, 50, 50)
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 47.8)
PAD = 3

# --- scene as measured from first_frame.png -------------------------------
A_ANCHOR = (468, 497)       # top-left ball of cluster A (grid origin)
A_BALLS = [(468, 497), (521, 497), (468, 550)]
CLUSTERS = {
    # name: (color, ball centers, label box (x0,y0,x1,y1))
    "B": ((50, 100, 255), [(192, 337), (192, 390)], (241, 359, 353, 401)),
    "C": ((255, 191, 0), [(482, 783), (534, 783), (482, 835), (534, 835)], (568, 805, 679, 848)),
    "D": ((75, 0, 130), [(218, 713), (271, 713), (218, 765), (271, 765), (218, 818)], (326, 762, 441, 804)),
}
A_LABEL_BOX = (555, 519, 668, 562)

# Absorption order (only clusters smaller than A may be absorbed):
# A=3 > B=2 -> 5 ; 5 > C=4 -> 9 ; 9 > D=5 -> 14
ORDER = ["B", "C", "D"]
# (flight_start, flight_end) frame indices per step; label of A updates at flight_end
STEPS = {"B": (8, 23), "C": (28, 45), "D": (50, 69)}


def layout(n):
    """Grid slots for n balls anchored at A's top-left ball."""
    cols = max(1, round(math.sqrt(n)))
    slots = []
    for i in range(n):
        r, c = divmod(i, cols)
        slots.append((A_ANCHOR[0] + c * SPACING, A_ANCHOR[1] + r * SPACING))
    return slots


def cluster_geom(slots):
    xs = [p[0] for p in slots]
    ys = [p[1] for p in slots]
    right = max(xs) + R
    cy = (min(ys) - R + max(ys) + R) / 2.0
    return right, cy


def label_pos(slots):
    right, cy = cluster_geom(slots)
    return right + 16, math.floor(cy - 10)


def smoothstep(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_color(c0, c1, t):
    return tuple(int(round(lerp(a, b, t))) for a, b in zip(c0, c1))


def draw_ball(d, cx, cy, color):
    cx, cy = int(round(cx)), int(round(cy))
    d.ellipse([cx + 1 - R, cy + 1 - R, cx + 1 + R, cy + 1 + R], fill=SHADOW)
    d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=color)


def draw_label(d, text, tx, ty):
    bb = d.textbbox((tx, ty), text, font=FONT)
    d.rectangle([bb[0] - PAD, bb[1] - PAD, bb[2] + PAD, bb[3] + PAD], fill=(255, 255, 255))
    d.text((tx, ty), text, font=FONT, fill=(0, 0, 0))


def main():
    base = Image.open(FIRST).convert("RGB")
    base_np = np.array(base)

    # Static plate: original frame with every dynamic element erased.
    plate = base.copy()
    pd = ImageDraw.Draw(plate)
    for cx, cy in A_BALLS:
        pd.rectangle([cx - R, cy - R, cx + R + 1, cy + R + 1], fill=BG)
    for color, balls, box in CLUSTERS.values():
        for cx, cy in balls:
            pd.rectangle([cx - R, cy - R, cx + R + 1, cy + R + 1], fill=BG)
        pd.rectangle(list(box), fill=BG)
    pd.rectangle(list(A_LABEL_BOX), fill=BG)
    plate_np = np.array(plate)

    # Ball state: list of dicts with source pos/color, target pos, step
    balls = [dict(pos=p, color=RED, src=p, dst=p, step=None) for p in A_BALLS]
    count = len(A_BALLS)
    for name in ORDER:
        color, members, _ = CLUSTERS[name]
        new_n = count + len(members)
        slots = layout(new_n)
        taken = {b["dst"] for b in balls}
        free = [s for s in slots if s not in taken]
        # existing grids are always subsets of the larger grid, so free slots
        # are exactly the new ones; pair sources to slots by reading order
        free.sort(key=lambda p: (p[1], p[0]))
        srcs = sorted(members, key=lambda p: (p[1], p[0]))
        for s, t in zip(srcs, free):
            balls.append(dict(pos=s, color=color, src=s, dst=t, step=name, src_color=color))
        count = new_n

    counts_after = {}
    c = len(A_BALLS)
    for name in ORDER:
        c += len(CLUSTERS[name][1])
        counts_after[name] = c

    os.makedirs(OUT_DIR, exist_ok=True)
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
         "-r", str(FPS), OUT],
        stdin=subprocess.PIPE,
    )

    for f in range(N_FRAMES):
        if f == 0:
            frame = base.copy()   # exact first frame
        else:
            frame = Image.fromarray(plate_np.copy())
            d = ImageDraw.Draw(frame)

            # labels of clusters being / not yet absorbed (fade during flight)
            for name in ORDER:
                s0, s1 = STEPS[name]
                _, _, box = CLUSTERS[name]
                if f < s0:
                    alpha = 1.0
                elif f >= s1:
                    alpha = 0.0
                else:
                    alpha = 1.0 - smoothstep((f - s0) / (s1 - s0))
                if alpha > 0:
                    x0, y0, x1, y1 = box
                    crop = base_np[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
                    bgc = np.array(BG, np.float32)
                    blend = (crop * alpha + bgc * (1 - alpha)).round().astype(np.uint8)
                    frame.paste(Image.fromarray(blend), (x0, y0))
                d = ImageDraw.Draw(frame)

            # current A count and label geometry
            a_count = len(A_BALLS)
            for name in ORDER:
                if f >= STEPS[name][1]:
                    a_count = counts_after[name]
            # label position interpolates from old to new layout during flight
            cur_step = None
            for name in ORDER:
                s0, s1 = STEPS[name]
                if s0 <= f < s1:
                    cur_step = name
            if cur_step is None:
                lx, ly = label_pos(layout(a_count))
            else:
                s0, s1 = STEPS[cur_step]
                t = smoothstep((f - s0) / (s1 - s0))
                p0 = label_pos(layout(a_count))
                p1 = label_pos(layout(counts_after[cur_step]))
                lx = int(round(lerp(p0[0], p1[0], t)))
                ly = int(round(lerp(p0[1], p1[1], t)))

            # balls: A's own balls first, then others (moving ones drawn on top)
            moving = []
            for b in balls:
                if b["step"] is None:
                    draw_ball(d, *b["pos"], RED)
                    continue
                s0, s1 = STEPS[b["step"]]
                if f < s0:
                    draw_ball(d, *b["src"], b["src_color"])
                elif f >= s1:
                    draw_ball(d, *b["dst"], RED)
                else:
                    t = smoothstep((f - s0) / (s1 - s0))
                    x = lerp(b["src"][0], b["dst"][0], t)
                    y = lerp(b["src"][1], b["dst"][1], t)
                    # colour turns red mostly in the second half of the flight
                    ct = smoothstep((t - 0.25) / 0.75)
                    moving.append((x, y, lerp_color(b["src_color"], RED, ct)))
            for x, y, col in moving:
                draw_ball(d, x, y, col)

            draw_label(d, f"A: {a_count}", lx, ly)

        ffmpeg.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())

    ffmpeg.stdin.close()
    ffmpeg.wait()
    if ffmpeg.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
