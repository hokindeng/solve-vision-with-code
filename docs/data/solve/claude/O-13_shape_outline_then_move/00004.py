#!/usr/bin/env python3
"""Generate the analogy video: the minus sign becomes outline-style, then moves down.

Style analysis of first_frame.png (top row, square A -> B -> C):
  * "filled" style  = PIL rectangle outline, width 2  (A: [107,261,269,421])
  * "outline" style = PIL rectangle outline, width 3, expanded 1px in y (B: [395,260,557,422])
  * step 2 moves the shape 60 px straight down (B -> C).
The minus (bottom row) is rect [107,662,269,702] width 2 -> outline [107,661,269,703] width 3
-> moved down to [107,721,269,763].
"""
import os, subprocess, tempfile
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT = os.path.join(HERE, "output", "video.mp4")
FPS, N = 16, 64
COLOR = (30, 91, 153)
SS = 4  # supersampling factor for sub-pixel motion

# minus geometry (filled style) and derived outline style
MX0, MY0, MX1, MY1 = 107, 662, 269, 702
OX0, OY0, OX1, OY1 = MX0, MY0 - 1, MX1, MY1 + 1
DROP = 60
# bounding box that we are allowed to touch (minus start + travel path), with margin
BX0, BY0, BX1, BY1 = MX0 - 3, OY0 - 3, MX1 + 4, OY1 + DROP + 4


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)  # smoothstep


def render_outline(bg, dy):
    """bg (H,W,3 uint8) + outline-style minus shifted down by dy (float px)."""
    h, w = bg.shape[:2]
    x0, y0 = BX0, BY0
    pw, ph = BX1 - BX0 + 1, BY1 - BY0 + 1
    big = Image.new("RGB", (pw * SS, ph * SS), (255, 255, 255))
    d = ImageDraw.Draw(big)
    oy = int(round(dy * SS))
    d.rectangle(
        [(OX0 - x0) * SS, (OY0 - y0) * SS + oy, (OX1 - x0) * SS + SS - 1, (OY1 - y0) * SS + SS - 1 + oy],
        outline=COLOR, width=3 * SS,
    )
    small = np.asarray(big.resize((pw, ph), Image.BOX)).astype(np.float32)
    out = bg.astype(np.float32).copy()
    patch = out[y0:y0 + ph, x0:x0 + pw]
    # multiply-blend onto background (background there is pure white anyway)
    out[y0:y0 + ph, x0:x0 + pw] = patch * (small / 255.0)
    return out


def main():
    first = np.asarray(Image.open(FIRST).convert("RGB"))
    bg = first.copy()
    bg[BY0:BY1 + 1, BX0:BX1 + 1] = 255  # erase the minus; everything else untouched
    filled_layer = first.astype(np.float32)
    outline_layer = render_outline(bg, 0.0)

    # timeline (frames)
    hold0, fade_end, hold1_end, move_end = 6, 30, 36, 60

    frames = []
    for i in range(N):
        if i < hold0:
            fr = first.astype(np.float32)
        elif i < fade_end:
            a = ease((i - hold0 + 1) / (fade_end - hold0))
            fr = (1 - a) * filled_layer + a * outline_layer
        elif i < hold1_end:
            fr = outline_layer
        elif i < move_end:
            t = ease((i - hold1_end + 1) / (move_end - hold1_end))
            fr = render_outline(bg, DROP * t)
        else:
            fr = render_outline(bg, DROP)
        frames.append(np.clip(fr + 0.5, 0, 255).astype(np.uint8))
    frames[0] = first.copy()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        for i, f in enumerate(frames):
            Image.fromarray(f).save(os.path.join(td, f"f{i:04d}.png"))
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
            "-i", os.path.join(td, "f%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
            "-r", str(FPS), OUT,
        ], check=True)
    print("wrote", OUT, len(frames), "frames")


if __name__ == "__main__":
    main()
