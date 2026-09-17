#!/usr/bin/env python3
"""Generate the analogy video: plus recolors to green (slot 2), then moves down (slot 3)."""
import os
import subprocess
import numpy as np
from PIL import Image

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 60, 16

base = np.array(Image.open(FIRST).convert("RGB")).astype(np.float32)
H, W, _ = base.shape

# --- scene constants measured from first_frame.png ---
BLUE = np.array([38, 63, 191], np.float32)
GREEN = np.array([95, 191, 111], np.float32)
COL_SHIFT = 287          # x distance between analogy slots
DOWN_SHIFT = 40          # y distance the star drops between slots 2 and 3
PLUS_BOX = (602, 763, 115, 276)          # y0, y1, x0, x1 (inclusive->exclusive)
Q_BOXES = [(655, 710, 465, 500), (655, 710, 752, 787)]  # "?" glyph regions (white elsewhere)

y0, y1, x0, x1 = PLUS_BOX
sprite = base[y0:y1, x0:x1].copy()
shape_mask = (sprite != 255).any(axis=2)                       # plus incl. outline
fill_mask = np.all(np.abs(sprite - BLUE) < 1, axis=2)          # blue fill only


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def erase_q(img, box, alpha):
    """Blend question-mark region toward white by alpha (0..1)."""
    a, b, c, d = box
    img[a:b, c:d] = img[a:b, c:d] * (1 - alpha) + 255 * alpha


def draw_plus(img, dx, dy, color, alpha=1.0):
    """Paste plus sprite shifted by (dx, dy) with given fill color and opacity."""
    spr = sprite.copy()
    spr[fill_mask] = color
    ty0, tx0 = y0 + dy, x0 + dx
    region = img[ty0:ty0 + spr.shape[0], tx0:tx0 + spr.shape[1]]
    m = shape_mask[..., None] * alpha
    region[:] = region * (1 - m) + spr * m


# timeline (frame indices)
P1_FADE = (2, 10)     # slot-2 "?" fades out, blue plus fades in
P1_COLOR = (10, 28)   # blue -> green
P2_FADE = (30, 38)    # slot-3 "?" fades out, green plus fades in
P2_MOVE = (38, 57)    # slide down by DOWN_SHIFT


def seg(f, a, b):
    return ease((f - a) / (b - a))


def render(f):
    img = base.copy()
    # ---- step 1: recolor at slot 2 ----
    if f >= P1_FADE[0]:
        a1 = seg(f, *P1_FADE)
        erase_q(img, Q_BOXES[0], a1)
        col = BLUE + (GREEN - BLUE) * seg(f, *P1_COLOR)
        draw_plus(img, COL_SHIFT, 0, col, a1)
    # ---- step 2: move down at slot 3 ----
    if f >= P2_FADE[0]:
        a2 = seg(f, *P2_FADE)
        erase_q(img, Q_BOXES[1], a2)
        dy = int(round(DOWN_SHIFT * seg(f, *P2_MOVE)))
        draw_plus(img, 2 * COL_SHIFT, dy, GREEN, a2)
    return np.clip(img + 0.5, 0, 255).astype(np.uint8)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = [render(f) for f in range(N_FRAMES)]
    assert np.array_equal(frames[0], base.astype(np.uint8))
    Image.fromarray(frames[-1]).save(os.path.join(OUT_DIR, "last_frame.png"))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
           "-movflags", "+faststart", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    p.wait()
    assert p.returncode == 0, "ffmpeg failed"
    print("wrote", OUT)


if __name__ == "__main__":
    main()
