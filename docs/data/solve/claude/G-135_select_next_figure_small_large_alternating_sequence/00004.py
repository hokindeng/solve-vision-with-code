#!/usr/bin/env python3
"""Regenerate /app/output/video.mp4: a red circle sweeps in around the correct
answer (option 4, the large dark-red circle) that continues the
small-large-small-large size pattern.  Every other pixel stays as in first_frame.png."""
import math, os, subprocess
import numpy as np
from PIL import Image, ImageDraw

BASE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(BASE, "first_frame.png")
OUT_DIR = os.path.join(BASE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

W = H = 1024
FPS = 16
N_FRAMES = 60
SS = 4  # supersampling factor for anti-aliased stroke

# Correct option: card 4 (x 751..961, y 752..956) -> center
CX, CY = 856.0, 854.5
RADIUS = 94.0
STROKE = 7.0
RED = (220, 20, 20)

HOLD_START = 8      # frames showing the unchanged first frame
SWEEP_END = 50      # frame at which the circle is complete


def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * t)


def circle_layer(frac):
    """RGBA layer (full-res) with a red arc covering `frac` of the circle."""
    if frac <= 0:
        return None
    big = Image.new("L", (W * SS, H * SS), 0)
    d = ImageDraw.Draw(big)
    r, s = RADIUS * SS, STROKE * SS
    box = [CX * SS - r, CY * SS - r, CX * SS + r, CY * SS + r]
    start = -90
    if frac >= 1:
        d.ellipse(box, outline=255, width=int(round(s)))
    else:
        d.arc(box, start, start + 360 * frac, fill=255, width=int(round(s)))
        # round caps
        for ang in (start, start + 360 * frac):
            a = math.radians(ang)
            px, py = CX * SS + (r - s / 2) * math.cos(a), CY * SS + (r - s / 2) * math.sin(a)
            d.ellipse([px - s / 2, py - s / 2, px + s / 2, py + s / 2], fill=255)
    mask = big.resize((W, H), Image.LANCZOS)
    return mask


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = Image.open(FIRST).convert("RGB")
    red = Image.new("RGB", (W, H), RED)
    frames = []
    for i in range(N_FRAMES):
        if i < HOLD_START:
            frac = 0.0
        elif i >= SWEEP_END:
            frac = 1.0
        else:
            frac = ease((i - HOLD_START) / (SWEEP_END - HOLD_START))
        mask = circle_layer(frac)
        frame = base if mask is None else Image.composite(red, base, mask)
        frames.append(np.asarray(frame.convert("RGB"), dtype=np.uint8))

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "medium",
           "-movflags", "+faststart", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print("wrote", OUT, len(frames), "frames")


if __name__ == "__main__":
    main()
