#!/usr/bin/env python3
"""Ball-eating puzzle video generator.

Reads /app/first_frame.png, detects the black ball and the colored balls,
finds the order in which the black ball can eat every colored ball (it may
only eat balls smaller than itself and grows after each meal), and renders
the animation to /app/output/video.mp4 (1024x1024, 16 fps, 108 frames).

Every pixel that the task does not require changing is copied verbatim from
the first frame: untouched balls and the background are never redrawn.
"""
import os
import subprocess
import numpy as np
import cv2
from PIL import Image

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

FPS = 16
N_FRAMES = 108
GROWTH = 0.75          # radius gained = GROWTH * radius of eaten ball
TRAVEL, EAT, HOLD = 19, 6, 2   # frames per stage; 4 targets * 27 = 108


def detect_balls(img):
    """Return list of dicts {cx, cy, r, color, mask} for each disk."""
    bg = img[0, 0].astype(int)
    diff = np.abs(img.astype(int) - bg).sum(2)
    mask = (diff > 30).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    balls = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 50:
            continue
        m = lab == i
        ys, xs = np.where(m)
        # median colour of the component
        col = np.median(img[ys, xs], axis=0).astype(int)
        balls.append(dict(cx=float(cent[i][0]), cy=float(cent[i][1]),
                          r=float(np.sqrt(area / np.pi)), color=tuple(col),
                          mask=m))
    return balls


def find_sequence(black_r, targets):
    """Greedy smallest-first order; verify the black ball can eat each one."""
    order = sorted(range(len(targets)), key=lambda i: targets[i]["r"])
    r = black_r
    for i in order:
        if r <= targets[i]["r"]:
            raise RuntimeError("no feasible eating sequence")
        r += GROWTH * targets[i]["r"]
    return order


def ease(t):
    return t * t * (3 - 2 * t)  # smoothstep


def clamp_center(cx, cy, r, w, h, margin=2):
    cx = min(max(cx, r + margin), w - r - margin)
    cy = min(max(cy, r + margin), h - r - margin)
    return cx, cy


def draw_ball(frame, cx, cy, r, color):
    shift = 4
    f = 1 << shift
    cv2.circle(frame, (int(round(cx * f)), int(round(cy * f))),
               int(round(r * f)), tuple(int(c) for c in color), -1,
               lineType=cv2.LINE_AA, shift=shift)


def main():
    img = np.array(Image.open(FIRST).convert("RGB"))
    h, w = img.shape[:2]
    bg_color = tuple(int(c) for c in img[0, 0])
    balls = detect_balls(img)
    black = min(balls, key=lambda b: sum(b["color"]))
    targets = [b for b in balls if b is not black]
    order = find_sequence(black["r"], targets)

    # Build per-frame state: black ball (cx, cy, r), set of eaten targets.
    states = []
    cx, cy, r = black["cx"], black["cy"], black["r"]
    eaten = set()
    states.append((cx, cy, r, frozenset(eaten)))  # frame 0 == first frame
    for idx in order:
        tgt = targets[idx]
        sx, sy, sr = cx, cy, r
        tx, ty = tgt["cx"], tgt["cy"]
        r_new = r + GROWTH * tgt["r"]
        fx, fy = clamp_center(tx, ty, r_new, w, h)
        # travel to the target
        for k in range(1, TRAVEL + 1):
            a = ease(k / TRAVEL)
            states.append((sx + (tx - sx) * a, sy + (ty - sy) * a, sr,
                           frozenset(eaten)))
        # eat: grow and settle inside the canvas
        eaten.add(idx)
        for k in range(1, EAT + 1):
            a = ease(k / EAT)
            states.append((tx + (fx - tx) * a, ty + (fy - ty) * a,
                           sr + (r_new - sr) * a, frozenset(eaten)))
        for _ in range(HOLD):
            states.append((fx, fy, r_new, frozenset(eaten)))
        cx, cy, r = fx, fy, r_new
    states = states[:N_FRAMES]
    while len(states) < N_FRAMES:
        states.append(states[-1])

    os.makedirs(OUT_DIR, exist_ok=True)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
         "-preset", "slow", "-r", str(FPS), OUT],
        stdin=subprocess.PIPE)
    for i, (bx, by, br, eaten) in enumerate(states):
        if i == 0:
            frame = img.copy()
        else:
            frame = img.copy()
            frame[black["mask"]] = bg_color       # erase original black ball
            for idx in eaten:                     # erase eaten balls
                frame[targets[idx]["mask"]] = bg_color
            draw_ball(frame, bx, by, br, black["color"])
        ff.stdin.write(np.ascontiguousarray(frame).tobytes())
    ff.stdin.close()
    ff.wait()
    if ff.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print("wrote", OUT, "frames:", len(states),
          "order:", [targets[i]["color"] for i in order])


if __name__ == "__main__":
    main()
