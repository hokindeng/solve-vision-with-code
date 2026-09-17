#!/usr/bin/env python3
"""Regenerate /app/output/video.mp4: the bottom-left outline pentagon is
scaled down (step 1) and then converted between outline and fill (step 2).
Everything else in the frame is copied verbatim from first_frame.png."""
import math
import subprocess
import numpy as np
from PIL import Image, ImageDraw

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
N_FRAMES, FPS = 16, 16

GREEN = (89, 153, 76)
CX, CY = 173.0, 682.0        # pentagon centre (cell centre)
R = 50.0                     # circumradius of the stroke centre-line
STROKE = 6.0
SCALE_END = 0.75             # ratio taken from the top example row
BBOX = (100, 600, 260, 760)  # region containing only the pentagon


def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * t)


def pent(cx, cy, r):
    return [(cx + r * math.sin(2 * math.pi * k / 5),
             cy - r * math.cos(2 * math.pi * k / 5)) for k in range(5)]


def draw_pentagon(base, r, inner_frac):
    """Draw a pentagon of circumradius r; inner_frac=1 -> outline of STROKE width,
    inner_frac=0 -> fully filled."""
    off = (STROKE / 2) / math.cos(math.pi / 5)      # edge offset -> radius offset
    r_out = r + off
    r_in = max((r - off) * inner_frac, 0.0)
    ss = 4                                           # supersample for smooth edges
    w, h = base.size
    mask = Image.new("L", (w * ss, h * ss), 0)
    d = ImageDraw.Draw(mask)
    d.polygon([(x * ss, y * ss) for x, y in pent(CX, CY, r_out)], fill=255)
    if r_in > 0.5:
        d.polygon([(x * ss, y * ss) for x, y in pent(CX, CY, r_in)], fill=0)
    mask = mask.resize((w, h), Image.LANCZOS)
    layer = Image.new("RGB", (w, h), GREEN)
    out = base.copy()
    out.paste(layer, (0, 0), mask)
    return out


def main():
    first = Image.open(SRC).convert("RGB")
    arr = np.array(first)
    # background with the pentagon erased (only green + white inside BBOX)
    x0, y0, x1, y1 = BBOX
    sub = arr[y0:y1, x0:x1]
    sub[np.all(sub == GREEN, axis=2)] = 255
    bg = Image.fromarray(arr)

    frames = [first]
    half = N_FRAMES // 2
    for i in range(1, N_FRAMES):
        if i <= half:                       # step 1: scale change
            t = ease(i / half)
            s = 1.0 + (SCALE_END - 1.0) * t
            frames.append(draw_pentagon(bg, R * s, 1.0))
        else:                               # step 2: outline <-> fill conversion
            t = ease((i - half) / (N_FRAMES - 1 - half))
            frames.append(draw_pentagon(bg, R * SCALE_END, 1.0 - t))

    for k, f in enumerate(frames):
        f.save(f"/tmp/frame_{k:03d}.png")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", "/tmp/frame_%03d.png", "-frames:v", str(N_FRAMES),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-qp", "0", "-preset", "slow",
        OUT], check=True)
    print("wrote", OUT, len(frames), "frames")


if __name__ == "__main__":
    main()
