#!/usr/bin/env python3
"""Generate video: mark the correct option (4th card, large yellow pentagon)
with a red circle that sweeps in over time. All other pixels stay identical
to first_frame.png in every frame."""
import subprocess, math
import numpy as np
from PIL import Image, ImageDraw

BASE = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
W = H = 1024
FPS = 16
N_FRAMES = 60

# Correct answer: option 4 (pattern small-large-small -> large). Card spans
# x 751..961, y 752..956; shape centered at ~(856, 854).
CX, CY = 856.0, 854.0
RADIUS = 92.0
STROKE = 7.0
COLOR = (230, 30, 30)
SS = 4  # supersampling factor for anti-aliased circle

START, END = 8, 50  # frames over which the arc sweeps 0 -> 360 degrees


def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * t)


def circle_layer(sweep_deg):
    """Return (rgb, alpha) of a red arc covering sweep_deg degrees, starting
    at the top and going clockwise."""
    if sweep_deg <= 0:
        return None
    big = Image.new("L", (W * SS, H * SS), 0)
    d = ImageDraw.Draw(big)
    r_o = (RADIUS + STROKE / 2) * SS
    box = [CX * SS - r_o, CY * SS - r_o, CX * SS + r_o, CY * SS + r_o]
    start = -90.0
    if sweep_deg >= 360:
        d.ellipse(box, outline=255, width=int(STROKE * SS))
    else:
        d.arc(box, start=start, end=start + sweep_deg, fill=255,
              width=int(STROKE * SS))
        # round caps
        cap_r = STROKE * SS / 2
        for ang in (start, start + sweep_deg):
            a = math.radians(ang)
            px = CX * SS + RADIUS * SS * math.cos(a)
            py = CY * SS + RADIUS * SS * math.sin(a)
            d.ellipse([px - cap_r, py - cap_r, px + cap_r, py + cap_r], fill=255)
    alpha = np.asarray(big.resize((W, H), Image.LANCZOS)).astype(np.float32) / 255.0
    return alpha


def main():
    base = np.asarray(Image.open(BASE).convert("RGB")).astype(np.float32)
    color = np.array(COLOR, dtype=np.float32).reshape(1, 1, 3)

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
           "-preset", "slow", "-r", str(FPS), OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    for i in range(N_FRAMES):
        if i < START:
            sweep = 0.0
        elif i >= END:
            sweep = 360.0
        else:
            sweep = 360.0 * ease((i - START) / (END - START))
        frame = base
        alpha = circle_layer(sweep)
        if alpha is not None:
            a = alpha[..., None]
            frame = base * (1 - a) + color * a
        proc.stdin.write(np.clip(frame + 0.5, 0, 255).astype(np.uint8).tobytes())

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
