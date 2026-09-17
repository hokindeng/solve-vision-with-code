#!/usr/bin/env python3
"""Bouncing-ball video: ball falls from 12.5 m (v0 = 0.9 m/s down), g = 10.1, e = 0.70."""
import os, subprocess, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(ROOT, "first_frame.png")
OUT_DIR = os.path.join(ROOT, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# --- physics ---
H0, V0, G, E = 12.5, -0.9, 10.1, 0.70
V_STOP = 0.5           # bounce speed below which the ball comes to rest

# --- scene geometry measured from first_frame.png ---
FPS, N_FRAMES = 16, 192
PX_PER_M = 32.0
BALL_X = 435
BALL_CY0 = 337         # ball centre at h = 12.5 m
BALL_R = 35            # incl. outline
GREEN = (60, 180, 60)
ARROW_PX_PER_MPS = 6.0 # same scale as the g arrow (10.1 -> 61 px)
ARROW_MIN = 16
SHAFT_HW, HEAD_HW, HEAD_LEN, GAP = 2, 6, 13, 5

first = Image.open(FIRST).convert("RGB")
# ball sprite (region is pure white apart from the ball)
sprite = first.crop((BALL_X - BALL_R, BALL_CY0 - BALL_R, BALL_X + BALL_R + 1, BALL_CY0 + BALL_R + 1))
# background: erase ball + arrow + label
bg = first.copy()
ImageDraw.Draw(bg).rectangle((396, 298, 645, 400), fill=(255, 255, 255))
font = ImageFont.truetype(FONT, 30)


def simulate(n_frames, fps):
    """Return list of (h, v) per frame, with exact bounce handling."""
    h, v = H0, V0
    out = []
    dt = 1.0 / fps
    resting = False
    for i in range(n_frames):
        out.append((h, v))
        if resting:
            continue
        t = dt
        while t > 1e-12:
            # time to reach ground within remaining t (h + v s - 0.5 g s^2 = 0)
            disc = v * v + 2 * G * h
            s_hit = (v + math.sqrt(max(disc, 0.0))) / G if disc >= 0 else float("inf")
            if s_hit <= t and s_hit >= 0:
                v_hit = v - G * s_hit
                h, v = 0.0, -E * v_hit
                t -= s_hit
                if v < V_STOP:
                    h, v, resting = 0.0, 0.0, True
                    break
            else:
                h = h + v * t - 0.5 * G * t * t
                v = v - G * t
                t = 0.0
        h = max(h, 0.0)
    return out


def cy_of(h):
    return BALL_CY0 - (h - H0) * PX_PER_M


def draw_arrow(d, x, y0, length, down):
    """Crisp arrow like the original: shaft 5 px wide, 13-row head, blunt 5 px tip."""
    sgn = 1 if down else -1
    tip = y0 + sgn * (length - 1)
    base = tip - sgn * (HEAD_LEN - 1)
    a, b = sorted((y0, base))
    d.rectangle((x - SHAFT_HW, a, x + SHAFT_HW, b), fill=GREEN)
    for k in range(HEAD_LEN):
        hw = max(SHAFT_HW, HEAD_HW - (k + 1) // 2)
        y = base + sgn * k
        d.line([(x - hw, y), (x + hw, y)], fill=GREEN)


GROUND_Y = 772         # last white row above the ground line


def render(h, v):
    im = bg.copy()
    cy = int(round(cy_of(h)))
    im.paste(sprite, (BALL_X - BALL_R, cy - BALL_R))
    d = ImageDraw.Draw(im)
    speed = abs(v)
    ty = cy + 28                      # label beside the downward arrow ...
    if ty + 31 > GROUND_Y:            # ... unless it would touch the ground
        ty = cy - 58
    if speed > 0.05:
        down = v < 0
        L = max(ARROW_MIN, int(round(speed * ARROW_PX_PER_MPS)))
        if down:
            y0 = cy + BALL_R + GAP
            L = min(L, GROUND_Y - y0 + 1)   # never draw over the ground
            if L >= HEAD_LEN:
                draw_arrow(d, BALL_X, y0, L, True)
        else:
            y0 = cy - BALL_R - GAP
            draw_arrow(d, BALL_X, y0, L, False)
            ty = cy - 58
    d.text((BALL_X + 25, ty), f"v={speed:.1f} m/s", font=font, fill=GREEN)
    return im


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    states = simulate(N_FRAMES, FPS)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "1024x1024", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "medium", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i, (h, v) in enumerate(states):
        im = first if i == 0 else render(h, v)
        p.stdin.write(np.asarray(im, dtype=np.uint8).tobytes())
    p.stdin.close()
    p.wait()
    print("wrote", OUT, "frames:", len(states))


if __name__ == "__main__":
    main()
