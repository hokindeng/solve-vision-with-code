#!/usr/bin/env python3
"""Generate /app/output/video.mp4: step-by-step solution of the large/small
alternation puzzle shown in /app/first_frame.png.

Sequence (top):   SMALL, LARGE, SMALL  ->  next must be LARGE
Choices (bottom): 1 red large triangle (wrong color), 2 pink circle (wrong shape),
                  3 pink large triangle (correct), 4 pink small triangle (wrong size)
"""
import os
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

FPS = 16
N_FRAMES = 60
SS = 2  # supersampling factor for anti-aliased overlay drawing

# --- geometry measured from first_frame.png -------------------------------
SEQ_ITEMS = [  # (cx, cy, half-size, label)
    (241, 337, 22, "SMALL"),
    (421, 337, 54, "LARGE"),
    (602, 337, 22, "SMALL"),
]
DASH_BOX = (717, 272, 847, 402)  # x0, y0, x1, y1
CARDS = [(63, 754, 269, 954), (293, 754, 499, 954), (523, 754, 729, 954), (753, 754, 959, 954)]
CARD_CX = [166, 396, 626, 856]
CARD_CY = 854
CORRECT = 2  # index of option 3
VERDICTS = ["wrong color", "wrong shape", "match", "wrong size"]

RED = (220, 40, 40, 255)
GREEN = (40, 150, 70, 255)
BLUE = (40, 90, 200, 255)
DARK = (60, 60, 60, 255)

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size * SS)
    except OSError:
        return ImageFont.load_default()


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def seg(frame, start, dur):
    """Progress in [0,1] of an action starting at `start` lasting `dur` frames."""
    return ease((frame - start) / float(dur))


def text_center(d, xy, s, f, fill):
    x, y = xy
    bbox = d.textbbox((0, 0), s, font=f)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text((x * SS - w / 2 - bbox[0], y * SS - h / 2 - bbox[1]), s, font=f, fill=fill)


def ring(d, cx, cy, r, color, width, sweep=1.0):
    box = [(cx - r) * SS, (cy - r) * SS, (cx + r) * SS, (cy + r) * SS]
    if sweep >= 1.0:
        d.ellipse(box, outline=color, width=width * SS)
    elif sweep > 0:
        d.arc(box, start=-90, end=-90 + 360 * sweep, fill=color, width=width * SS)


def draw_overlay(frame):
    """Return an RGBA overlay (1024x1024) with the annotations for this frame."""
    W = 1024 * SS
    ov = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    f_label = font(20)
    f_small = font(16)
    f_big = font(26)

    # ---- Step 1 (frames 2..20): label each sequence item in turn ---------
    for i, (cx, cy, hs, lab) in enumerate(SEQ_ITEMS):
        p = seg(frame, 2 + 6 * i, 5)
        if p <= 0:
            continue
        a = int(255 * p)
        col = (*BLUE[:3], a)
        ring(d, cx, cy, hs + 14, col, 3, sweep=p)
        if p > 0.5:
            text_center(d, (cx, cy - hs - 36), lab, f_label, (*BLUE[:3], int(255 * (p - 0.5) * 2)))

    # ---- Step 1b (frames 20..28): deduce next size -------------------------
    p = seg(frame, 20, 7)
    if p > 0:
        a = int(255 * p)
        x0, y0, x1, y1 = DASH_BOX
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        text_center(d, (cx, y0 - 36), "LARGE?", f_label, (*BLUE[:3], a))
        text_center(d, (512, 470), "Pattern: SMALL, LARGE, SMALL  ->  next is LARGE", f_big, (*DARK[:3], a))
        # Ghost outline of the expected answer: a large triangle in the box
        hs = 54
        pts = [((cx) * SS, (cy - hs) * SS), ((cx - hs) * SS, (cy + hs) * SS), ((cx + hs) * SS, (cy + hs) * SS)]
        d.polygon(pts, outline=(211, 54, 130, int(180 * p)), width=3 * SS)

    # ---- Step 2 (frames 28..46): check each option --------------------------
    for i in range(4):
        p = seg(frame, 28 + 4 * i, 4)
        if p <= 0:
            continue
        a = int(255 * p)
        x0, y0, x1, y1 = CARDS[i]
        ok = i == CORRECT
        col = (*(GREEN if ok else RED)[:3], a)
        mark = "OK" if ok else "X"
        text_center(d, (CARD_CX[i], y0 + 20), (mark + "  " + VERDICTS[i]) if not ok else "OK  match", f_small, col)

    # ---- Step 3 (frames 46..57): red circle sweeps around the answer -------
    p = seg(frame, 46, 11)
    if p > 0:
        ring(d, CARD_CX[CORRECT], CARD_CY, 112, RED, 6, sweep=p)

    return ov.resize((1024, 1024), Image.LANCZOS)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = Image.open(FIRST).convert("RGB")
    frames = []
    for k in range(N_FRAMES):
        if k == 0:
            frames.append(np.array(base))
            continue
        ov = draw_overlay(k)
        fr = Image.alpha_composite(base.convert("RGBA"), ov).convert("RGB")
        frames.append(np.array(fr))

    raw = os.path.join(OUT_DIR, "_frames.rgb")
    with open(raw, "wb") as fh:
        for fr in frames:
            fh.write(np.ascontiguousarray(fr, dtype=np.uint8).tobytes())
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1024x1024", "-r", str(FPS),
        "-i", raw,
        "-c:v", "libx264", "-preset", "slow", "-crf", "12", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", OUT,
    ]
    subprocess.run(cmd, check=True)
    os.remove(raw)
    print("wrote", OUT, "frames:", len(frames))


if __name__ == "__main__":
    main()
