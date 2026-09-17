#!/usr/bin/env python3
"""Generate video: mark the correct option (large blue square, option 1) with a red circle."""
import subprocess
import numpy as np
from PIL import Image, ImageDraw

BASE = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
W = H = 1024
FPS = 16
N_FRAMES = 60

# Option 1 box spans x 63..269, y 754..954 -> center (166, 854).
CX, CY = 166, 854
RADIUS = 108
STROKE = 7
RED = (220, 30, 30)

HOLD_START = 8        # frames of unchanged first frame
DRAW_FRAMES = 36      # frames for sweeping the circle
SS = 4                # supersampling factor for anti-aliasing


def circle_frame(base, sweep):
    """Return base with a red arc covering `sweep` (0..1) of the full circle."""
    if sweep <= 0:
        return base.copy()
    pad = RADIUS + STROKE + 4
    x0, y0 = CX - pad, CY - pad
    x1, y1 = CX + pad, CY + pad
    region = base.crop((x0, y0, x1, y1))
    big = region.resize((region.width * SS, region.height * SS), Image.LANCZOS)
    d = ImageDraw.Draw(big)
    r = RADIUS * SS
    c = pad * SS
    bbox = (c - r, c - r, c + r, c + r)
    start = -90
    end = -90 + 360 * min(sweep, 1.0)
    if sweep >= 1.0:
        d.ellipse(bbox, outline=RED, width=STROKE * SS)
    else:
        d.arc(bbox, start=start, end=end, fill=RED, width=STROKE * SS)
        # round caps
        cap = STROKE * SS / 2
        for ang in (start, end):
            a = np.deg2rad(ang)
            px = c + r * np.cos(a) - cap
            py = c + r * np.sin(a) - cap
            d.ellipse((px, py, px + 2 * cap, py + 2 * cap), fill=RED)
    small = big.resize(region.size, Image.LANCZOS)
    out = base.copy()
    out.paste(small, (x0, y0))
    return out


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)


def main():
    base = Image.open(BASE).convert("RGB")
    assert base.size == (W, H)
    frames = []
    for i in range(N_FRAMES):
        if i < HOLD_START:
            sweep = 0.0
        elif i < HOLD_START + DRAW_FRAMES:
            sweep = ease((i - HOLD_START + 1) / DRAW_FRAMES)
        else:
            sweep = 1.0
        frames.append(circle_frame(base, sweep))

    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
        "-r", str(FPS), OUT,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for f in frames:
        p.stdin.write(np.asarray(f, dtype=np.uint8).tobytes())
    p.stdin.close()
    p.wait()
    frames[-1].save("/app/output/last_frame.png")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
