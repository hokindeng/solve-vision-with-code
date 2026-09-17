#!/usr/bin/env python3
"""Render a bouncing-ball video (gravity 6.9 m/s^2, e=0.70) on top of first_frame.png."""
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W = H = 1024
FPS = 16
N_FRAMES = 192
DURATION = N_FRAMES / FPS

# --- physics parameters (from prompt) ---
H0 = 15.0        # m
V0 = -1.8        # m/s (downward)
G = 6.9          # m/s^2
E = 0.70         # coefficient of restitution
V_STOP = 0.25    # rebound speed below which the ball is considered at rest

# --- geometry measured from first_frame.png ---
PX_PER_M = 32.0
GROUND_Y = 773           # first row of the ground line
BALL_X = 267
BALL_R = 27
BALL_FILL = (220, 180, 60)
BALL_EDGE = (0, 0, 0)
ARROW_COL = (60, 180, 60)
TRAIL_COL = (235, 205, 130)
ARROW_PX_PER_MPS = 16.0 / 1.8   # 1.8 m/s arrow is 16 px long in the first frame
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)


def ball_center_y(h):
    return GROUND_Y - BALL_R - h * PX_PER_M


def simulate():
    """Analytic piecewise simulation; returns a function state(t) -> (h, v) and the stop time."""
    segments = []  # (t_start, h_start, v_start)
    t, h, v = 0.0, H0, V0
    while True:
        segments.append((t, h, v))
        # time to hit ground: h + v*dt - 0.5*G*dt^2 = 0
        disc = v * v + 2 * G * h
        dt = (v + np.sqrt(disc)) / G
        t += dt
        v_impact = v - G * dt
        v = -E * v_impact
        h = 0.0
        if v < V_STOP:
            break
    t_stop = t

    def state(tt):
        if tt >= t_stop:
            return 0.0, 0.0
        idx = 0
        for i, seg in enumerate(segments):
            if seg[0] <= tt:
                idx = i
        t0, h0, v0 = segments[idx]
        d = tt - t0
        return max(0.0, h0 + v0 * d - 0.5 * G * d * d), v0 - G * d

    return state, t_stop


def draw_arrow(draw, cx, y_from, y_to):
    """Vertical arrow from y_from to y_to (tip). Shaft width 5, triangular head."""
    L = abs(y_to - y_from)
    if L < 1:
        return
    sign = 1 if y_to > y_from else -1
    head = min(L, max(12.0, 0.3 * L))
    hw = max(2.5, 0.5 * head)
    y_base = y_to - sign * head
    draw.rectangle([cx - 2, min(y_from, y_to), cx + 2, max(y_from, y_to)], fill=ARROW_COL)
    draw.polygon([(cx - hw, y_base), (cx + hw, y_base), (cx, y_to)], fill=ARROW_COL)


def render_frame(bg, h, v, trail):
    im = bg.copy()
    d = ImageDraw.Draw(im)
    cy = ball_center_y(h)
    for ty in trail:
        d.ellipse([BALL_X - 3, ty - 3, BALL_X + 3, ty + 3], fill=TRAIL_COL)
    cyr = int(round(cy))
    d.ellipse([BALL_X - BALL_R - 1, cyr - BALL_R - 1, BALL_X + BALL_R + 1, cyr + BALL_R + 1],
              fill=BALL_FILL, outline=BALL_EDGE, width=2)
    speed = abs(v)
    label = f"v={speed:.1f} m/s"
    if speed > 0.05:
        L = speed * ARROW_PX_PER_MPS
        if v < 0:   # moving down: arrow below the ball
            y_from = cyr + BALL_R + 6
            y_to = y_from + L - 1
        else:       # moving up: arrow above the ball
            y_from = cyr - BALL_R - 6
            y_to = y_from - L + 1
        draw_arrow(d, BALL_X, y_from, y_to)
        mid = (y_from + y_to) / 2.0
        baseline = int(round(mid - 4 + 11.5))
    else:
        baseline = cyr + 11
    # keep the label fully above the ground line
    baseline = min(baseline, GROUND_Y - 4)
    text_x = BALL_X + 25
    if baseline - 22 < cyr + BALL_R - 5 and baseline > cyr - BALL_R + 5:
        text_x = BALL_X + BALL_R + 8   # label sits beside the ball, not over it
    d.text((text_x, baseline), label, font=FONT, fill=ARROW_COL, anchor="ls")
    # nothing may alter the ground: restore it from the background
    im.paste(bg.crop((0, GROUND_Y, W, H)), (0, GROUND_Y))
    return im


def main():
    first = Image.open("/app/first_frame.png").convert("RGB")
    bg = first.copy()
    # Remove the ball, its velocity arrow and the label; everything else stays untouched.
    ImageDraw.Draw(bg).rectangle([236, 235, 470, 320], fill=(255, 255, 255))

    state, t_stop = simulate()
    # Keep real-time pacing unless the ball would not settle before the video ends.
    time_scale = min(1.0, (DURATION - 0.6) / t_stop)

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "15", "-pix_fmt", "yuv420p",
           "-r", str(FPS), "/app/output/video.mp4"]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    trail = []
    for i in range(N_FRAMES):
        t = i / FPS * time_scale
        h, v = state(t)
        frame = render_frame(bg, h, v, trail)
        trail.append(int(round(ball_center_y(h))))
        if i == 0:
            frame.save("/app/output/frame_000.png")
        if i == N_FRAMES - 1:
            frame.save("/app/output/frame_last.png")
        proc.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"stop time {t_stop:.2f}s, time_scale {time_scale:.3f}")


if __name__ == "__main__":
    main()
