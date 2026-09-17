#!/usr/bin/env python3
"""Draw a red circle step by step around the largest number (96) in first_frame.png."""
import os, subprocess, shutil, tempfile
import numpy as np
from PIL import Image, ImageDraw

BASE = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS, N_FRAMES, SS = 16, 80, 4          # SS = supersample factor for anti-aliasing

# ---- locate the numbers and pick the largest -------------------------------
NUMBERS = {  # value -> bounding box (x0, y0, x1, y1), measured from first_frame.png
    32: (108, 253, 262, 350),
    92: (326, 184, 475, 274),
    96: (640, 203, 793, 293),
    51: (408, 790, 560, 880),
}
target = max(NUMBERS)                   # numerical comparison -> 96
x0, y0, x1, y1 = NUMBERS[target]
cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
half_diag = np.hypot(x1 - x0, y1 - y0) / 2
RADIUS = half_diag + 20                 # ~109 px, clears the digits with a margin
WIDTH = 7
RED = (220, 20, 20)

base = np.array(Image.open(BASE).convert("RGB")).astype(np.float32)
H, W = base.shape[:2]

# Timeline: hold -> draw arc -> hold
HOLD_START, DRAW_END = 16, 68           # frames 16..67 draw the circle, 68..79 hold


def arc_mask(sweep_deg):
    """Anti-aliased coverage mask (0..1) of a red arc from -90deg sweeping clockwise."""
    if sweep_deg <= 0:
        return np.zeros((H, W), np.float32)
    big = Image.new("L", (W * SS, H * SS), 0)
    d = ImageDraw.Draw(big)
    bbox = [(cx - RADIUS) * SS, (cy - RADIUS) * SS, (cx + RADIUS) * SS, (cy + RADIUS) * SS]
    start = -90
    if sweep_deg >= 360:
        d.ellipse(bbox, outline=255, width=WIDTH * SS)
    else:
        d.arc(bbox, start, start + sweep_deg, fill=255, width=WIDTH * SS)
        # round pen caps at both ends
        r = WIDTH * SS / 2
        for ang in (start, start + sweep_deg):
            a = np.deg2rad(ang)
            px, py = (cx + RADIUS * np.cos(a)) * SS, (cy + RADIUS * np.sin(a)) * SS
            d.ellipse([px - r, py - r, px + r, py + r], fill=255)
    small = big.resize((W, H), Image.LANCZOS)
    return np.asarray(small, np.float32) / 255.0


def ease(t):
    return t * t * (3 - 2 * t)          # smoothstep


def make_frame(i):
    if i < HOLD_START:
        sweep = 0.0
    elif i >= DRAW_END:
        sweep = 360.0
    else:
        sweep = 360.0 * ease((i - HOLD_START) / (DRAW_END - 1 - HOLD_START))
    m = arc_mask(sweep)[..., None]
    out = base * (1 - m) + np.array(RED, np.float32) * m
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = tempfile.mkdtemp()
    for i in range(N_FRAMES):
        Image.fromarray(make_frame(i)).save(f"{tmp}/f{i:04d}.png")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", f"{tmp}/f%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "12", "-preset", "slow", "-r", str(FPS), OUT], check=True)
    shutil.rmtree(tmp)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
