#!/usr/bin/env python3
"""Render the bouncing-ball video described in /app/prompt.txt.

Scene geometry is measured from /app/first_frame.png:
  * ground surface at y = 775 px, 32 px per metre (ticks every 5 m = 160 px)
  * ball: centre x = 727, radius 36 px (2 px black outline, green fill)
  * velocity arrow (green) along the ball's vertical axis, label right-aligned
    to the left of the arrow and vertically centred on it.
Everything else in the frame is left untouched.
"""
import os
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

# ---- physics (from prompt) -------------------------------------------------
H0 = 13.3        # m
V0 = 0.7         # m/s, upward positive
G = 6.1          # m/s^2
E = 0.70         # coefficient of restitution
V_STOP = 0.3     # m/s: rebound speed below which the ball is considered at rest

# ---- video -----------------------------------------------------------------
FPS = 16
N_FRAMES = 192
W = H = 1024

# ---- scene -----------------------------------------------------------------
GROUND_Y = 775          # first row of ground surface
PX_PER_M = 32.0
BALL_X = 727
BALL_R = 36
BALL_FILL = (60, 200, 80)
BALL_OUTLINE = (0, 0, 0)
ARROW_COLOR = (60, 180, 60)
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
SHAFT_W = 5
ARROW_GAP = 5           # gap between ball surface and arrow base
LABEL_DX = 25           # label right edge sits this far left of the arrow axis


def simulate(n_frames, fps):
    """Return list of (height, velocity) for each frame, exact piecewise-analytic."""
    dt = 1.0 / fps
    h, v = H0, V0
    stopped = False
    states = []
    for _ in range(n_frames):
        states.append((h, 0.0 if stopped else v))
        if stopped:
            continue
        rem = dt
        while rem > 1e-12:
            # time to reach ground: h + v t - g t^2 / 2 = 0, smallest positive root
            disc = v * v + 2 * G * h
            t_hit = (v + np.sqrt(disc)) / G if disc >= 0 else np.inf
            if t_hit <= rem:
                v_imp = v - G * t_hit
                rem -= t_hit
                h = 0.0
                v = -E * v_imp
                if v < V_STOP:
                    stopped, v = True, 0.0
                    break
            else:
                h = h + v * rem - 0.5 * G * rem * rem
                v = v - G * rem
                rem = 0.0
        if h < 0:
            h = 0.0
    return states


def make_base():
    """First frame with the dynamic elements (ball, arrow, label) erased."""
    img = Image.open(FIRST).convert("RGB")
    d = ImageDraw.Draw(img)
    d.rectangle([BALL_X - BALL_R - 1, 276, BALL_X + BALL_R + 1, 350], fill=(255, 255, 255))
    d.rectangle([530, 248, 745, 283], fill=(255, 255, 255))
    return img


def ball_center_y(h):
    return int(round(GROUND_Y - h * PX_PER_M - BALL_R))


def draw_frame(base, h, v):
    img = base.copy()
    d = ImageDraw.Draw(img)
    cy = ball_center_y(h)
    speed = abs(v)

    # velocity arrow
    if speed > 0.05:
        L = 6.0 + speed * 14.0                      # px, tip to base
        head_len = int(round(min(max(0.35 * L, 7.0), 22.0)))
        head_half = head_len - 1
        if v > 0:   # moving up
            base_y = cy - BALL_R - ARROW_GAP
            tip_y = base_y - L
            sgn = -1
        else:       # moving down
            base_y = cy + BALL_R + ARROW_GAP
            tip_y = base_y + L
            sgn = 1
        base_y_i, tip_y_i = int(round(base_y)), int(round(tip_y))
        # shaft (5 px wide, centred on BALL_X)
        y0, y1 = sorted((base_y_i, tip_y_i))
        d.rectangle([BALL_X - SHAFT_W // 2, y0, BALL_X + SHAFT_W // 2, y1], fill=ARROW_COLOR)
        # head (filled triangle, apex at the tip)
        hb = tip_y_i - sgn * head_len
        d.polygon([(BALL_X, tip_y_i), (BALL_X - head_half, hb), (BALL_X + head_half, hb)],
                  fill=ARROW_COLOR)
        mid_y = (base_y_i + tip_y_i) / 2.0
    else:
        mid_y = cy - BALL_R - ARROW_GAP - 8

    # label, right-aligned left of arrow axis, vertically centred on arrow
    label = f"v={speed:.1f} m/s"
    d.text((BALL_X - LABEL_DX, mid_y - 1.5), label, font=FONT, fill=ARROW_COLOR, anchor="rm")

    # ball (drawn last so it sits on top)
    d.ellipse([BALL_X - BALL_R, cy - BALL_R, BALL_X + BALL_R, cy + BALL_R],
              fill=BALL_FILL, outline=BALL_OUTLINE, width=2)
    return img


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = make_base()
    first = Image.open(FIRST).convert("RGB")
    states = simulate(N_FRAMES, FPS)

    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "16",
           "-r", str(FPS), OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i, (h, v) in enumerate(states):
        frame = first if i == 0 else draw_frame(base, h, v)
        proc.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {OUT} ({len(states)} frames @ {FPS} fps)")


if __name__ == "__main__":
    main()
