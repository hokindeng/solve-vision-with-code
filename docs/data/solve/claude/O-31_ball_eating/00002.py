#!/usr/bin/env python3
"""Black ball eats the colored balls in the only feasible order (smallest first)."""
import os, subprocess, tempfile
import numpy as np
from PIL import Image, ImageDraw

W = H = 1024
FPS = 16
N_FRAMES = 95
OUT = "/app/output/video.mp4"
FIRST = "/app/first_frame.png"

BG = (255, 255, 255)
# (cx, cy, r, color) as measured from first_frame.png
BLACK = [678.0, 320.0, 28.0]
BALLS = [
    dict(c=(917.0, 309.0), r=38.0, col=(255, 105, 180)),  # pink
    dict(c=(955.0, 655.0), r=58.0, col=(60, 179, 113)),   # green
    dict(c=(97.0, 696.0),  r=20.0, col=(255, 140, 0)),    # orange
]
GROW = 0.6  # r_new = r + GROW * r_eaten


def plan():
    """Greedy: always eat the largest ball smaller than the black ball."""
    r = BLACK[2]
    remaining = list(range(len(BALLS)))
    order = []
    while remaining:
        cand = [i for i in remaining if BALLS[i]["r"] < r]
        if not cand:
            raise RuntimeError("no feasible sequence")
        i = max(cand, key=lambda k: BALLS[k]["r"])
        order.append(i)
        r += GROW * BALLS[i]["r"]
        remaining.remove(i)
    return order


def ease(t):
    return t * t * (3 - 2 * t)


def draw(black, balls_state):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    for b in balls_state:
        if b["r"] <= 0:
            continue
        cx, cy = b["c"]
        r = round(b["r"])
        d.ellipse([round(cx) - r, round(cy) - r, round(cx) + r, round(cy) + r], fill=b["col"])
    cx, cy, r = black
    r = round(r)
    d.ellipse([round(cx) - r, round(cy) - r, round(cx) + r, round(cy) + r], fill=(0, 0, 0))
    return im


def main():
    order = plan()
    balls = [dict(c=b["c"], r=b["r"], col=b["col"]) for b in BALLS]
    black = list(BLACK)

    # time budget: split the action frames across the steps, keep a short hold at the end
    hold_end = 6
    action = N_FRAMES - 1 - hold_end
    per = action // len(order)
    frames = []
    frames.append(draw(black, balls))

    for step, idx in enumerate(order):
        tgt = balls[idx]
        n_step = per if step < len(order) - 1 else action - per * (len(order) - 1)
        n_absorb = max(5, n_step // 4)
        n_move = n_step - n_absorb
        sx, sy, r0 = black
        tx, ty = tgt["c"]
        r_eat = tgt["r"]
        r1 = r0 + GROW * r_eat
        for k in range(1, n_move + 1):
            t = ease(k / n_move)
            black = [sx + (tx - sx) * t, sy + (ty - sy) * t, r0]
            frames.append(draw(black, balls))
        for k in range(1, n_absorb + 1):
            t = ease(k / n_absorb)
            tgt["r"] = r_eat * (1 - t)
            black = [tx, ty, r0 + (r1 - r0) * t]
            frames.append(draw(black, balls))
        tgt["r"] = 0.0
        black = [tx, ty, r1]

    while len(frames) < N_FRAMES:
        frames.append(draw(black, balls))
    frames = frames[:N_FRAMES]

    # guarantee frame 0 is exactly the reference
    frames[0] = Image.open(FIRST).convert("RGB")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        for i, f in enumerate(frames):
            f.save(os.path.join(td, f"f{i:04d}.png"))
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
            "-i", os.path.join(td, "f%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
            "-r", str(FPS), OUT,
        ], check=True)
    print("order:", [BALLS[i]["col"] for i in order], "->", OUT, len(frames), "frames")


if __name__ == "__main__":
    main()
