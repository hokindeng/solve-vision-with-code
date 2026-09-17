#!/usr/bin/env python3
"""Generate the step-by-step solution video for the large/small alternation task.

Sequence (top): small hexagon, LARGE hexagon, small hexagon, ?  ->  next must be LARGE.
Choices (bottom): 1 small dark hexagon, 2 large dark circle, 3 large cyan hexagon,
                  4 LARGE dark hexagon  -> option 4 is correct.
"""
import math
import os
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(ROOT, "first_frame.png")
OUT = os.path.join(ROOT, "output", "video.mp4")
FPS = 16
N_FRAMES = 60
W = H = 1024

# Geometry measured from first_frame.png
SEQ_ITEMS = [(241, 337, "SMALL"), (420, 337, "LARGE"), (601, 337, "SMALL")]
QBOX = (717, 271, 847, 401)          # dashed "?" box
OPTION_CENTERS = [(166, 854), (396, 854), (626, 854), (856, 854)]
CORRECT = 3                          # index of the correct option (option 4)
CIRCLE_R = 84
RED = (220, 30, 30)
DARK = (35, 35, 35)


def font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


F_LABEL = font(22)
F_NEXT = font(26)


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * t)


def draw_center_text(d, xy, text, fill, f):
    x, y = xy
    bbox = d.textbbox((0, 0), text, font=f)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text((x - w / 2 - bbox[0], y - h / 2 - bbox[1]), text, fill=fill, font=f)


def render_frame(base, i):
    """Return frame i as a PIL image. Frame 0 is exactly the first frame."""
    img = base.copy()
    d = ImageDraw.Draw(img)

    # Step 1 (frames 1..24): label each sequence item SMALL / LARGE / SMALL in turn.
    for k, (cx, cy, label) in enumerate(SEQ_ITEMS):
        start = 2 + k * 7
        if i >= start:
            draw_center_text(d, (cx, 215), label, DARK, F_LABEL)

    # Step 2 (frames 24..36): announce the deduced next size inside the "?" box.
    if i >= 24:
        qx = (QBOX[0] + QBOX[2]) // 2
        draw_center_text(d, (qx, 215), "LARGE", RED, F_LABEL)
        draw_center_text(d, (qx, QBOX[3] + 22), "next = LARGE", RED, F_NEXT)

    # Step 3 (frames 34..56): sweep a red circle around the correct option.
    if i >= 34:
        t = ease((i - 34) / 22.0)
        cx, cy = OPTION_CENTERS[CORRECT]
        box = (cx - CIRCLE_R, cy - CIRCLE_R, cx + CIRCLE_R, cy + CIRCLE_R)
        # supersample the arc for smooth edges
        S = 4
        layer = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        sbox = tuple(v * S for v in box)
        end = -90 + 360 * t
        if t >= 0.999:
            ld.ellipse(sbox, outline=RED + (255,), width=6 * S)
        else:
            ld.arc(sbox, start=-90, end=end, fill=RED + (255,), width=6 * S)
        layer = layer.resize((W, H), Image.LANCZOS)
        img = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")
    return img


def main():
    base = Image.open(FIRST).convert("RGB")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(N_FRAMES):
            fr = render_frame(base, i)
            if i == 0:
                assert np.array_equal(np.array(fr), np.array(base))
            fr.save(os.path.join(tmp, f"f{i:04d}.png"))
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
            "-i", os.path.join(tmp, "f%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
            "-r", str(FPS), OUT,
        ], check=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
