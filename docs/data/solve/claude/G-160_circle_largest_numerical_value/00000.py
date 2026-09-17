#!/usr/bin/env python3
"""Draw one red circle, step by step, around the largest number (97) in first_frame.png."""
import os, subprocess, shutil, tempfile
import numpy as np
from PIL import Image, ImageDraw

BASE = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
W = H = 1024
FPS = 16
N_FRAMES = 80

# Numbers in scene: 77, 1, 17, 51, 97, 64 -> largest is 97.
# Bounding box of "97" glyphs measured from the frame: x 593-743, y 637-727.
CX, CY = 668.0, 682.0
R = 96.0
STROKE = 7
COLOR = (220, 30, 30, 255)
SS = 4  # supersampling factor for anti-aliasing

START, END = 14, 68  # frames during which the circle is drawn


def circle_overlay(fraction):
    """RGBA overlay (1024x1024) with a red arc covering `fraction` of the circle."""
    ov = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    if fraction <= 0:
        return ov.resize((W, H), Image.LANCZOS)
    d = ImageDraw.Draw(ov)
    bbox = [(CX - R) * SS, (CY - R) * SS, (CX + R) * SS, (CY + R) * SS]
    start = -90.0  # begin at the top, draw clockwise
    end = start + 360.0 * min(fraction, 1.0)
    if fraction >= 1.0:
        d.ellipse(bbox, outline=COLOR, width=STROKE * SS)
    else:
        d.arc(bbox, start=start, end=end, fill=COLOR, width=STROKE * SS)
        # round caps so the growing line looks like a pen stroke
        r_cap = STROKE * SS / 2
        for ang in (start, end):
            a = np.deg2rad(ang)
            px, py = (CX + R * np.cos(a)) * SS, (CY + R * np.sin(a)) * SS
            d.ellipse([px - r_cap, py - r_cap, px + r_cap, py + r_cap], fill=COLOR)
    return ov.resize((W, H), Image.LANCZOS)


def main():
    base = Image.open(BASE).convert("RGBA")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = tempfile.mkdtemp()
    for i in range(N_FRAMES):
        if i < START:
            frac = 0.0
        elif i >= END:
            frac = 1.0
        else:
            t = (i - START) / (END - START)
            frac = t * t * (3 - 2 * t)  # ease in/out
        frame = base if frac <= 0 else Image.alpha_composite(base, circle_overlay(frac))
        frame.convert("RGB").save(os.path.join(tmp, f"f_{i:04d}.png"))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp, "f_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT,
    ], check=True)
    shutil.rmtree(tmp)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
