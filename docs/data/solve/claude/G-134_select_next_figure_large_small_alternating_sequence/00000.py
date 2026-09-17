#!/usr/bin/env python3
"""Generate the step-by-step solution video for the large/small alternation puzzle.

Frame 0 is exactly first_frame.png.  The animation then:
  1. inspects each sequence shape left to right and labels it SMALL / LARGE,
  2. states the pattern and the required next size (LARGE),
  3. scans the four options, rejecting wrong ones,
  4. draws a red circle around the correct option (option 4).
All drawing happens only in the empty white regions plus the red circle; every
other pixel is left as in the first frame.
"""
import os
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N = 16, 60

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
if not os.path.exists(FONT_PATH):
    FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def font(sz):
    try:
        return ImageFont.truetype(FONT_PATH, sz)
    except Exception:
        return ImageFont.load_default()


# Geometry measured from first_frame.png
SEQ = [  # (cx, cy, radius, label)
    (241, 337, 34, "SMALL"),
    (421, 337, 54, "LARGE"),
    (602, 337, 34, "SMALL"),
]
QBOX = (717, 271, 847, 401)          # dashed "?" box
OPTS = [(166, 854), (396, 854), (626, 854), (856, 854)]   # option centres
OPT_BOX_HALF = 103
CORRECT = 3
RED = (220, 30, 30)
GRAY = (120, 120, 120)
GREEN = (30, 150, 60)

base = Image.open(FIRST).convert("RGB")


def ease(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def text_center(d, xy, s, f, fill):
    w = d.textlength(s, font=f)
    d.text((xy[0] - w / 2, xy[1]), s, font=f, fill=fill)


def render(i):
    im = base.copy()
    d = ImageDraw.Draw(im)
    f_small = font(22)
    f_med = font(26)

    # ---- Step 1: inspect sequence items (frames 1..21, 7 frames each)
    for k, (cx, cy, r, lab) in enumerate(SEQ):
        start = 1 + 7 * k
        if i >= start:
            # brief highlight ring while inspecting this item
            if i < start + 7:
                d.ellipse([cx - r - 8, cy - r - 8, cx + r + 8, cy + r + 8],
                          outline=GRAY, width=3)
            text_center(d, (cx, 415), lab, f_small, GRAY)

    # ---- Step 2: state the pattern and the next size (frames 22..33)
    if i >= 22:
        text_center(d, (512, 150), "Pattern: SMALL, LARGE, SMALL, ...", f_med, GRAY)
    if i >= 27:
        text_center(d, (512, 195), "Next must be: LARGE", f_med, RED)
        text_center(d, (782, 415), "LARGE", f_small, RED)
    if i >= 30:
        text_center(d, (512, 660), "Same shape (circle), same color, LARGE size", f_small, GRAY)

    # ---- Step 3: scan options (frames 34..45, 3 frames per option)
    if i >= 34:
        for k, (cx, cy) in enumerate(OPTS):
            start = 34 + 3 * k
            if i < start:
                break
            if k != CORRECT:
                # rejected option: thin gray outline + cross mark above box
                if i < start + 3:
                    d.rectangle([cx - OPT_BOX_HALF - 6, cy - OPT_BOX_HALF - 6,
                                 cx + OPT_BOX_HALF + 6, cy + OPT_BOX_HALF + 6],
                                outline=GRAY, width=3)
                reasons = ["small", "wrong color", "wrong shape"]
                text_center(d, (cx, 982), "X  " + reasons[k], f_small, GRAY)
            else:
                text_center(d, (cx, 982), "OK  LARGE circle", f_small, GREEN)

    # ---- Step 4: draw the red circle around option 4 (frames 44..59)
    if i >= 44:
        t = ease((i - 44) / 14.0)
        cx, cy = OPTS[CORRECT]
        R = 114
        bbox = [cx - R, cy - R, cx + R, cy + R]
        if t >= 1.0:
            d.ellipse(bbox, outline=RED, width=7)
        else:
            d.arc(bbox, start=-90, end=-90 + 360 * t, fill=RED, width=7)
    return im


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = [np.array(render(i)) for i in range(N)]
    assert np.array_equal(frames[0], np.array(base))
    raw = os.path.join(OUT_DIR, "frames.raw")
    with open(raw, "wb") as fh:
        for fr in frames:
            fh.write(fr.tobytes())
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "1024x1024", "-r", str(FPS), "-i", raw,
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "medium",
           "-r", str(FPS), OUT]
    subprocess.run(cmd, check=True)
    os.remove(raw)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
