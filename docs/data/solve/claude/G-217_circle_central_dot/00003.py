#!/usr/bin/env python3
"""Circle the middle-by-count dot with a red circle, animated over 22 frames."""
import os
import subprocess
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES = 22
FPS = 16
SS = 4  # supersampling factor for antialiased circle


def find_dots(img):
    a = np.array(img.convert("L")) < 128
    lab, n = ndimage.label(a)
    dots = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        dots.append((xs.mean(), ys.mean(), (xs.max() - xs.min() + 1) / 2.0))
    dots.sort(key=lambda d: d[0])  # left to right
    return dots


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = Image.open(FIRST).convert("RGB")
    W, H = base.size
    dots = find_dots(base)
    cx, cy, r = dots[len(dots) // 2]  # equal count on each side
    spacing = min(dots[i + 1][0] - dots[i][0] for i in range(len(dots) - 1)) if len(dots) > 1 else 4 * r
    width = 3
    radius = min(r + 4, (spacing - r) - width / 2 - 2)  # stay clear of neighbours

    frames = []
    for i in range(N_FRAMES):
        frame = base.copy()
        if i > 0:
            t = i / (N_FRAMES - 1)
            sweep = 360.0 * t
            # Draw the arc on a supersampled transparent layer, then composite.
            layer = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
            d = ImageDraw.Draw(layer)
            box = [(cx - radius) * SS, (cy - radius) * SS, (cx + radius) * SS, (cy + radius) * SS]
            start = -90.0
            d.arc(box, start, start + sweep, fill=(255, 0, 0, 255), width=int(width * SS))
            layer = layer.resize((W, H), Image.LANCZOS)
            frame = Image.alpha_composite(frame.convert("RGBA"), layer).convert("RGB")
        frames.append(frame)

    raw = b"".join(np.asarray(f, dtype=np.uint8).tobytes() for f in frames)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-qp", "0", "-preset", "slow",
        "-r", str(FPS), OUT,
    ]
    subprocess.run(cmd, input=raw, check=True)
    print(f"wrote {OUT}: {len(frames)} frames, circle at ({cx:.0f},{cy:.0f}) r={radius:.1f}")


if __name__ == "__main__":
    main()
