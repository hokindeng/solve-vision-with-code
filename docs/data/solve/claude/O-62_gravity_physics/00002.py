#!/usr/bin/env python3
"""Render the bouncing-ball video described in /app/prompt.txt.

The scene (ground, height scale, gravity arrow/label) is taken verbatim from
first_frame.png; only the ball, its velocity arrow and the velocity label are
redrawn each frame.
"""
import math
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W = H = 1024
FPS = 16
DURATION = 12.0
N_FRAMES = int(FPS * DURATION)

# --- physics parameters (from prompt) ---
H0 = 11.2      # m
V0 = 3.6       # m/s upward
G = 10.0       # m/s^2
E = 0.70       # coefficient of restitution
V_STOP = 0.5   # m/s: below this after a bounce the ball is considered at rest

# --- scene geometry measured from first_frame.png ---
GROUND_Y = 773.0        # pixel y of ground surface (top row of ground line, 0 m)
PX_PER_M = 32.0         # 20 m span = 640 px
BALL_X = 561            # ball centre column
BALL_R = 30             # outer radius incl. 2 px black outline
BALL_FILL = (220, 180, 60)
ARROW_COLOR = (60, 180, 60)
PX_PER_MPS = 6.0        # velocity arrow: 6 px per m/s (same as g arrow)
ARROW_GAP = 5           # gap between ball and arrow tail
SHAFT_W = 5
HEAD_LEN, HEAD_HALF_W = 12, 6
FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 30)

FIRST = Image.open('/app/first_frame.png').convert('RGB')


def build_background():
    """First frame with the ball, velocity arrow and label removed (all sit on white)."""
    bg = FIRST.copy()
    d = ImageDraw.Draw(bg)
    d.rectangle([BALL_X - BALL_R - 2, 355 - 2, BALL_X + BALL_R + 2, 415 + 2], fill=(255, 255, 255))
    d.rectangle([360, 318, 580, 356], fill=(255, 255, 255))
    return bg


def simulate_segments():
    """Return list of (t_start, t_end, h_start, v_start) parabolic segments and stop time."""
    segs = []
    t, h, v = 0.0, H0, V0
    while True:
        # time to hit ground: h + v*dt - 0.5*g*dt^2 = 0
        disc = v * v + 2 * G * h
        dt = (v + math.sqrt(disc)) / G
        segs.append((t, t + dt, h, v))
        v_impact = v - G * dt
        t += dt
        h = 0.0
        v = -E * v_impact
        if v < V_STOP:
            return segs, t


SEGMENTS, T_STOP = simulate_segments()


def state(t):
    """Height (m) and velocity (m/s, up positive) at time t."""
    if t >= T_STOP:
        return 0.0, 0.0
    for t0, t1, h0, v0 in SEGMENTS:
        if t <= t1:
            dt = t - t0
            return max(0.0, h0 + v0 * dt - 0.5 * G * dt * dt), v0 - G * dt
    return 0.0, 0.0


def draw_arrow(d, x, y_tail, y_tip):
    length = abs(y_tip - y_tail)
    if length < 1:
        return
    sgn = 1 if y_tip > y_tail else -1
    hl = min(HEAD_LEN, max(3, int(round(length * 0.55))))
    hw = max(3, int(round(HEAD_HALF_W * hl / HEAD_LEN)))
    d.line([(x, y_tail), (x, y_tip)], fill=ARROW_COLOR, width=SHAFT_W)
    base = y_tip - sgn * hl
    d.polygon([(x, y_tip), (x - hw, base), (x + hw, base)], fill=ARROW_COLOR)


def render(t, bg):
    h, v = state(t)
    img = bg.copy()
    d = ImageDraw.Draw(img)
    cy = int(round(GROUND_Y - h * PX_PER_M - BALL_R))
    # ball
    d.ellipse([BALL_X - BALL_R, cy - BALL_R, BALL_X + BALL_R, cy + BALL_R],
              fill=BALL_FILL, outline=(0, 0, 0), width=2)
    # velocity arrow + label
    label = f"v={abs(v):.1f} m/s"
    if abs(v) > 1e-6:
        length = abs(v) * PX_PER_MPS
        if v > 0:
            y_tail = cy - BALL_R - ARROW_GAP
            y_tip = int(round(y_tail - length))
        else:
            y_tail = cy + BALL_R + ARROW_GAP
            y_tip = int(round(y_tail + length))
        y_tip = min(max(y_tip, 5), int(GROUND_Y) - 1)
        draw_arrow(d, BALL_X, y_tail, y_tip)
        mid = (y_tail + y_tip) / 2.0
    else:
        mid = cy
    # label sits left of the arrow; if the label would overlap the ball (very
    # short or absent arrow) shift it fully clear of the ball.
    label_x = BALL_X - 25
    if abs(mid - cy) < BALL_R + 14:
        label_x = BALL_X - BALL_R - 8
    d.text((label_x, int(round(mid)) - 2), label, font=FONT, fill=ARROW_COLOR, anchor='rm')
    return img


def main():
    bg = build_background()
    frames = []
    for i in range(N_FRAMES):
        t = i / FPS
        frames.append(FIRST if i == 0 else render(t, bg))
    # write via ffmpeg
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '/app/output/video.mp4']
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.asarray(f, dtype=np.uint8).tobytes())
    p.stdin.close()
    p.wait()
    print(f"wrote {N_FRAMES} frames; ball stops at t={T_STOP:.2f}s")


if __name__ == '__main__':
    main()
