#!/usr/bin/env python3
"""Render a bouncing-ball video (gravity 6.0 m/s^2, elasticity 0.70) on top of
/app/first_frame.png, showing the trajectory trace and a live velocity arrow."""
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

APP = Path("/app")
FIRST = APP / "first_frame.png"
OUT = APP / "output" / "video.mp4"

W = H = 1024
FPS = 16
N_FRAMES = 192
DURATION = N_FRAMES / FPS  # 12.0 s

# ---- physics --------------------------------------------------------------
H0 = 20.2        # m, height of the ball's bottom above ground
V0 = 0.0         # m/s (upward positive)
G = 6.0          # m/s^2
E = 0.70         # coefficient of restitution
V_STOP = 0.5     # m/s: below this post-bounce speed the ball is considered at rest
HOLD_AT_END = 0.75  # s of real video with the ball resting at the end

# ---- scene geometry (measured from first_frame.png) -----------------------
PX_PER_M = 32.0
BALL_X = 716
BALL_R = 36
BALL_Y0 = 91                      # ball centre in the first frame (h = 20.2 m)
GROUND_Y = BALL_Y0 + BALL_R + H0 * PX_PER_M  # y of ball bottom at h = 0

BALL_FILL = (220, 180, 60)
BALL_OUTLINE = (0, 0, 0)
ARROW_COL = (200, 40, 40)
TRACE_COL = (235, 170, 110)
APEX_COL = (160, 110, 40)
V_SCALE = 8.0    # px per m/s for the velocity arrow

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def simulate(dt=1e-4):
    """Return arrays (t, h, v) plus apex list [(t, h), ...] until the ball rests."""
    t, h, v = 0.0, H0, V0
    ts, hs, vs, apexes = [t], [h], [v], []
    resting = False
    while not resting:
        v_new = v - G * dt
        h_new = h + 0.5 * (v + v_new) * dt
        if h_new <= 0.0:
            # exact time of impact within this step
            # solve h + v*tau - 0.5*G*tau^2 = 0 for tau in (0, dt]
            disc = v * v + 2 * G * h
            tau = (v + math.sqrt(max(disc, 0.0))) / G
            v_impact = v - G * tau
            v_after = -E * v_impact
            t += tau
            h = 0.0
            if v_after < V_STOP:
                v = 0.0
                resting = True
            else:
                v = v_after
        else:
            if v > 0 and v_new <= 0:
                apexes.append((t + dt, h_new))
            t, h, v = t + dt, h_new, v_new
        ts.append(t); hs.append(h); vs.append(v)
    return np.array(ts), np.array(hs), np.array(vs), apexes


def ball_center_y(h):
    return GROUND_Y - h * PX_PER_M - BALL_R


def draw_arrow(draw, x0, y0, x1, y1, col, width=5, head=16):
    """Straight arrow from (x0,y0) to (x1,y1) with a filled triangular head."""
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return
    ux, uy = dx / L, dy / L
    head = min(head, L)
    bx, by = x1 - ux * head, y1 - uy * head  # base of the head
    draw.line([(x0, y0), (bx, by)], fill=col, width=width)
    px, py = -uy, ux
    hw = head * 0.6
    draw.polygon([(x1, y1), (bx + px * hw, by + py * hw), (bx - px * hw, by - py * hw)], fill=col)


def main():
    base = Image.open(FIRST).convert("RGB")
    bg = base.copy()
    # erase the ball from the background (it sits entirely on white)
    ImageDraw.Draw(bg).rectangle(
        [BALL_X - BALL_R - 2, BALL_Y0 - BALL_R - 2, BALL_X + BALL_R + 2, BALL_Y0 + BALL_R + 2],
        fill=(255, 255, 255))

    ts, hs, vs, apexes = simulate()
    T_sim = ts[-1]
    # map the whole motion (until rest) onto the video minus a short hold
    time_scale = T_sim / (DURATION - HOLD_AT_END)

    font = ImageFont.truetype(FONT_BOLD, 26)
    font_small = ImageFont.truetype(FONT_BOLD, 20)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-r", str(FPS), str(OUT)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    y_min_seen = ball_center_y(H0)  # top of trajectory trace
    y_max_seen = ball_center_y(H0)
    for i in range(N_FRAMES):
        if i == 0:
            frame = base  # first frame is exactly the given image
        else:
            t_sim = min(i / FPS * time_scale, T_sim)
            k = int(np.searchsorted(ts, t_sim))
            k = min(k, len(ts) - 1)
            h, v = float(hs[k]), float(vs[k])
            cy = ball_center_y(h)
            y_min_seen = min(y_min_seen, cy)
            y_max_seen = max(y_max_seen, cy)

            frame = bg.copy()
            d = ImageDraw.Draw(frame)

            # trajectory trace: path covered so far (vertical line through ball centre)
            d.line([(BALL_X, y_min_seen), (BALL_X, y_max_seen)], fill=TRACE_COL, width=3)
            # apex markers reached so far, with the peak height
            for (ta, ha) in apexes:
                if ta <= t_sim:
                    ya = ball_center_y(ha)
                    d.line([(BALL_X - 14, ya), (BALL_X + 14, ya)], fill=APEX_COL, width=2)
                    if ha >= 1.5:  # label the larger peaks only (small ones would overlap)
                        d.text((BALL_X + 20, ya - 12), f"{ha:.1f}m", fill=APEX_COL, font=font_small)

            # ball
            d.ellipse([BALL_X - BALL_R, cy - BALL_R, BALL_X + BALL_R, cy + BALL_R],
                      fill=BALL_FILL, outline=BALL_OUTLINE, width=2)

            # velocity arrow (direction and magnitude) + label to the left
            speed = abs(v)
            if speed > 0.05:
                length = speed * V_SCALE
                sign = -1 if v > 0 else 1  # screen y grows downward
                x0, y0 = BALL_X, cy + sign * BALL_R          # start at the ball's edge
                x1, y1 = BALL_X, cy + sign * (BALL_R + length)
                draw_arrow(d, x0, y0, x1, y1, ARROW_COL)
                label_y = (y0 + y1) / 2
            else:
                label_y = cy
            label = f"v = {speed:.1f} m/s" + ("" if speed <= 0.05 else (" ↑" if v > 0 else " ↓"))
            bbox = d.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            lx = BALL_X - BALL_R - 20 - tw
            ly = min(max(label_y - th / 2, 4), GROUND_Y - th - 6)
            d.text((lx, ly), label, fill=ARROW_COL, font=font)

        proc.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())

    proc.stdin.close()
    proc.wait()
    print(f"wrote {OUT}  (sim {T_sim:.2f}s -> {DURATION - HOLD_AT_END:.2f}s video, "
          f"{len(apexes)} bounces shown)")


if __name__ == "__main__":
    main()
