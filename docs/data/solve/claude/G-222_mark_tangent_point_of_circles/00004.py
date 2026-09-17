#!/usr/bin/env python3
"""Animate drawing a black circle around the tangent point of the two touching circles."""
import subprocess, math
import numpy as np
from PIL import Image, ImageDraw

BASE = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
W = H = 1024
FPS = 16
N_FRAMES = 60

# Measured from first_frame.png: dark circle c=(511.5,248) r=132, blue circle c=(512,512) r=132.
# They are externally tangent; tangent point lies on the segment between centers.
C1 = np.array([511.5, 248.0]); R1 = 132.0
C2 = np.array([512.0, 512.0]); R2 = 132.0
T = C1 + (C2 - C1) * (R1 / (R1 + R2))      # tangent point ~ (512, 380)
RING_R = 34       # radius of the marker circle
RING_W = 5        # stroke width
SS = 4            # supersampling factor for anti-aliasing
DRAW_FRAMES = 52  # frames used to sweep the circle; remaining frames hold the result


def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * t)


def render_frame(base, progress):
    """progress in [0,1]: fraction of the ring's arc drawn (clockwise from top)."""
    if progress <= 0:
        return base.copy()
    # Draw the ring on a supersampled transparent layer, then alpha-composite.
    layer = Image.new("L", (W * SS, H * SS), 0)
    d = ImageDraw.Draw(layer)
    cx, cy = T * SS
    r = RING_R * SS
    bbox = [cx - r, cy - r, cx + r, cy + r]
    start = -90
    end = start + 360 * progress
    if progress >= 1:
        d.ellipse(bbox, outline=255, width=RING_W * SS)
    else:
        d.arc(bbox, start=start, end=end, fill=255, width=RING_W * SS)
        # round caps at the ends of the partial arc
        cap = RING_W * SS / 2
        for ang in (start, end):
            a = math.radians(ang)
            px, py = cx + (r - cap) * math.cos(a), cy + (r - cap) * math.sin(a)
            d.ellipse([px - cap, py - cap, px + cap, py + cap], fill=255)
    mask = layer.resize((W, H), Image.LANCZOS)
    black = Image.new("RGB", (W, H), (0, 0, 0))
    out = base.copy()
    out.paste(black, (0, 0), mask)
    return out


def main():
    base = Image.open(BASE).convert("RGB")
    frames = []
    for i in range(N_FRAMES):
        p = 0.0 if i == 0 else min(1.0, ease(min(1.0, i / (DRAW_FRAMES - 1))))
        frames.append(np.asarray(render_frame(base, p)))
    raw = np.stack(frames).astype(np.uint8).tobytes()
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "slow", OUT]
    subprocess.run(cmd, input=raw, check=True)
    print("wrote", OUT, N_FRAMES, "frames")


if __name__ == "__main__":
    main()
