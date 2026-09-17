#!/usr/bin/env python3
"""Two balls move toward each other at equal speed and merge at the midpoint.
Overlapping regions use additive (light) color mixing."""
import os, subprocess, shutil, tempfile
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES, SIZE = 16, 80, 1024

R, OUTLINE_W = 120, 2                       # measured from first_frame.png
BALLS = [((358, 403), (97, 56, 74)),        # (center, fill color)
         ((611, 722), (113, 186, 93))]


def find_background(ref):
    """Erase the balls (fill + outline) from the reference; everything else stays."""
    bg = ref.copy()
    for (cx, cy), _ in BALLS:
        yy, xx = np.ogrid[:SIZE, :SIZE]
        m = (xx - cx) ** 2 + (yy - cy) ** 2 <= (R + 1) ** 2
        bg[m] = 255
    return bg


def ball_masks(center):
    """Return (fill_mask, outline_mask) using the same rasterizer as the source frame."""
    cx, cy = int(round(center[0])), int(round(center[1]))
    im = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(im).ellipse([cx - R, cy - R, cx + R, cy + R],
                               fill=1, outline=2, width=OUTLINE_W)
    a = np.array(im)
    return a == 1, a == 2


def render(bg, centers):
    acc = np.zeros((SIZE, SIZE, 3), np.int32)
    any_fill = np.zeros((SIZE, SIZE), bool)
    any_outline = np.zeros((SIZE, SIZE), bool)
    for c, (_, col) in zip(centers, BALLS):
        fill, outline = ball_masks(c)
        acc[fill] += np.array(col, np.int32)     # additive light mixing
        any_fill |= fill
        any_outline |= outline
    frame = bg.copy()
    frame[any_fill] = np.clip(acc[any_fill], 0, 255).astype(np.uint8)
    # Outline is only visible where no ball's light covers it (black adds nothing).
    frame[any_outline & ~any_fill] = 0
    return frame


def main():
    ref = np.array(Image.open(FIRST).convert("RGB"))
    bg = find_background(ref)
    p0 = np.array(BALLS[0][0], float)
    p1 = np.array(BALLS[1][0], float)
    mid = (p0 + p1) / 2

    tmp = tempfile.mkdtemp()
    for i in range(N_FRAMES):
        s = i / (N_FRAMES - 1)                  # 0 -> 1, linear (same speed both balls)
        centers = [p0 + (mid - p0) * s, p1 + (mid - p1) * s]
        frame = render(bg, centers)
        if i == 0:
            assert (frame == ref).all(), "frame 0 must equal first_frame.png"
        Image.fromarray(frame).save(os.path.join(tmp, f"{i:04d}.png"))

    os.makedirs(OUT_DIR, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", os.path.join(tmp, "%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
                    "-preset", "slow", OUT], check=True)
    shutil.rmtree(tmp)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
