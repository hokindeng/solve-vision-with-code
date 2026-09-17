#!/usr/bin/env python3
"""Generate the A:B :: C:? video: the bottom-right arrow appears in outline style.

Row 1 shows a filled L -> outlined L (each primitive outlined with 8px strokes and
round joints).  Row 2 has a filled arrow, so the answer is the same arrow drawn in
outline style at the position of the '?'.  The '?' fades out while the outline is
traced progressively over the clip.
"""
import os
import subprocess
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

N_FRAMES = 60
FPS = 16
BLUE = (66, 108, 191)
STROKE = 8
RADIUS = STROKE // 2

# Bottom-left arrow primitives (measured from first_frame.png), shifted to the
# bottom-right cell by the same offset the first row uses (+704 px in x).
DX = 704
RECT = [(80 + DX, 728), (160 + DX, 728), (160 + DX, 808), (80 + DX, 808)]
TRI = [(160 + DX, 688), (240 + DX, 768), (160 + DX, 848)]
PRIMITIVES = [RECT, TRI]

# Bounding box of the '?' glyph (gray text) that gets replaced.
Q_BOX = (846, 740, 882, 795)  # x0, y0, x1, y1 (exclusive)


def segments():
    segs = []
    for poly in PRIMITIVES:
        n = len(poly)
        for i in range(n):
            segs.append((poly[i], poly[(i + 1) % n]))
    return segs


SEGS = segments()
SEG_LEN = [float(np.hypot(b[0] - a[0], b[1] - a[1])) for a, b in SEGS]
TOTAL_LEN = sum(SEG_LEN)


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)


def draw_outline(img, length):
    """Draw the first `length` pixels of the outline path onto a PIL image."""
    dr = ImageDraw.Draw(img)
    remaining = length
    for (a, b), L in zip(SEGS, SEG_LEN):
        if remaining <= 0:
            break
        if remaining >= L:
            end = b
        else:
            f = remaining / L
            end = (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
        dr.line([a, end], fill=BLUE, width=STROKE)
        for (x, y) in (a, end):
            dr.ellipse([x - RADIUS, y - RADIUS, x + RADIUS, y + RADIUS], fill=BLUE)
        remaining -= L


def make_frame(base, i):
    t = i / (N_FRAMES - 1)
    arr = base.copy()

    # Fade the '?' out over the first quarter of the clip.
    fade = ease(min(1.0, t / 0.25))
    x0, y0, x1, y1 = Q_BOX
    region = arr[y0:y1, x0:x1].astype(np.float32)
    region = region * (1 - fade) + 255.0 * fade
    arr[y0:y1, x0:x1] = np.round(region).astype(np.uint8)

    # Trace the outline from ~15% to 100% of the clip.
    prog = ease(np.clip((t - 0.15) / 0.85, 0.0, 1.0))
    img = Image.fromarray(arr)
    if prog > 0:
        draw_outline(img, prog * TOTAL_LEN)
    return np.array(img)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = np.array(Image.open(FIRST).convert("RGB"))
    h, w = base.shape[:2]

    frames = [make_frame(base, i) for i in range(N_FRAMES)]
    frames[0] = base.copy()  # first frame is exactly first_frame.png

    Image.fromarray(frames[-1]).save(os.path.join(OUT_DIR, "last_frame.png"))

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-preset", "slow", "-crf", "10",
        "-pix_fmt", "yuv420p", "-r", str(FPS), "-movflags", "+faststart",
        OUT,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {OUT} ({len(frames)} frames @ {FPS} fps)")


if __name__ == "__main__":
    main()
