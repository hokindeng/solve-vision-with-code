#!/usr/bin/env python3
"""Generate a video that draws a red circle around the Chinese character (美)."""
import os, subprocess
import numpy as np
from PIL import Image, ImageDraw

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 48

# Bounding box of 美 (measured from first_frame.png): x 525-647, y 641-766
CX, CY = 587.0, 703.5
RADIUS = 88
WIDTH = 6
RED = (255, 0, 0)
SS = 4  # supersampling factor for anti-aliased strokes

base = Image.open(FIRST).convert("RGB")
W, H = base.size


def make_frame(t):
    """t in [0,1]: fraction of the circle drawn (starts at top, clockwise)."""
    if t <= 0:
        return base.copy()
    # Draw arc on a supersampled transparent layer, then composite.
    layer = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    box = [(CX - RADIUS) * SS, (CY - RADIUS) * SS, (CX + RADIUS) * SS, (CY + RADIUS) * SS]
    start = -90
    end = start + 360 * t
    if t >= 1:
        d.ellipse(box, outline=RED + (255,), width=WIDTH * SS)
    else:
        d.arc(box, start=start, end=end, fill=RED + (255,), width=WIDTH * SS)
        # round caps
        r = WIDTH * SS / 2
        for ang in (start, end):
            a = np.deg2rad(ang)
            px = (CX + RADIUS * np.cos(a)) * SS
            py = (CY + RADIUS * np.sin(a)) * SS
            # arc width extends inward from radius; cap centre sits at RADIUS - WIDTH/2
            px = (CX + (RADIUS - WIDTH / 2) * np.cos(a)) * SS
            py = (CY + (RADIUS - WIDTH / 2) * np.sin(a)) * SS
            d.ellipse([px - r, py - r, px + r, py + r], fill=RED + (255,))
    layer = layer.resize((W, H), Image.LANCZOS)
    frame = base.copy()
    frame.paste(layer, (0, 0), layer)
    return frame


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    # Frame 0 is unchanged; hold briefly, draw the circle over the middle, hold at the end.
    hold_start, hold_end = 4, 6
    draw_n = N_FRAMES - hold_start - hold_end
    for i in range(N_FRAMES):
        if i < hold_start:
            t = 0.0
        elif i >= N_FRAMES - hold_end:
            t = 1.0
        else:
            u = (i - hold_start + 1) / draw_n
            t = 0.5 - 0.5 * np.cos(np.pi * u)  # ease in/out
        frames.append(np.asarray(make_frame(t)))

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
