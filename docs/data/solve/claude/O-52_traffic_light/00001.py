#!/usr/bin/env python3
"""Regenerate /app/output/video.mp4: four independent traffic lights at a crossroad.

Cycle per light: Red(4s) -> Yellow(4s) -> Green(4s) -> Yellow(4s) -> Red ...
Initial state (t=0):  N red 4s, S green 4s, E green 3s, W yellow 1s.
The video simulates 6 seconds of light behaviour, paced over the 7 s clip:
  frames   0..7   : hold initial state (0.5 s)
  frames   8..104 : simulation time 0 -> 6 s (1 sim second == 1 real second)
  frames 105..111 : hold final state
Only the lamp discs and the countdown digits are modified; every other pixel is
copied from first_frame.png.
"""
import math
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

FPS = 16
N_FRAMES = 112
HOLD_START = 8            # frames holding the initial state
SIM_SECONDS = 6
SIM_FRAMES = SIM_SECONDS * FPS  # 96 frames of simulation

COLORS = {"red": (255, 0, 0), "yellow": (255, 200, 0), "green": (0, 200, 0)}
PHASE_LEN = 4  # seconds for every phase

# Phase ring: index -> colour.  Yellow appears twice (after red, after green).
RING = ["red", "yellow", "green", "yellow"]

# Geometry measured from first_frame.png
LIGHTS = {
    #        disc centre   box top-left (outer 127x127 box, 2px outline)
    "N": {"disc": (512, 214), "box": (449, 274)},
    "S": {"disc": (512, 798), "box": (449, 858)},
    "W": {"disc": (220, 506), "box": (157, 566)},
    "E": {"disc": (804, 506), "box": (741, 566)},
}
DISC_R = 72          # search radius; mask = lamp-coloured pixels within it
BOX_IN = (2, 2, 125, 125)  # interior of the box relative to box top-left
FONT_SIZE = 102
TEXT_OFF = (-1.0, 1.5)     # anchor offset from interior centre (measured)

# Initial state: (phase index in RING, seconds remaining in that phase)
# West is yellow; the yellow that precedes red is assumed (classic "about to
# stop" yellow).  Either yellow yields the same final state after 6 s because
# red and green both last 4 s.
INIT = {
    "N": (0, 4),  # red, 4 s
    "S": (2, 4),  # green, 4 s
    "E": (2, 3),  # green, 3 s
    "W": (3, 1),  # yellow (-> red), 1 s
}


def state_at(light, t):
    """Return (colour, displayed countdown) at simulation time t (seconds)."""
    idx, remaining = INIT[light]
    # advance whole phases
    while t >= remaining - 1e-9:
        t -= remaining
        idx = (idx + 1) % len(RING)
        remaining = PHASE_LEN
    left = remaining - t
    return RING[idx], int(math.ceil(left - 1e-9))


def make_disc_mask(shape, center, r):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= r * r


def render_frame(base, masks, font, t):
    frame = base.copy()
    img = None
    for name, geo in LIGHTS.items():
        color, count = state_at(name, t)
        frame[masks[name]] = COLORS[color]
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    for name, geo in LIGHTS.items():
        color, count = state_at(name, t)
        bx, by = geo["box"]
        x0, y0, x1, y1 = bx + BOX_IN[0], by + BOX_IN[1], bx + BOX_IN[2], by + BOX_IN[3]
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=(255, 255, 255))
        cx = (x0 + x1) / 2 + TEXT_OFF[0]
        cy = (y0 + y1) / 2 + TEXT_OFF[1]
        draw.text((cx, cy), str(count), font=font, fill=(0, 0, 0), anchor="mm")
    return np.array(img)


def sim_time(frame_idx):
    s = (frame_idx - HOLD_START) / FPS
    return min(max(s, 0.0), float(SIM_SECONDS))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = np.array(Image.open(FIRST).convert("RGB"))
    h, w = base.shape[:2]
    # Disc masks: only pixels that carry the flat fill colour inside radius 65.
    masks = {}
    for name, geo in LIGHTS.items():
        m = make_disc_mask((h, w), geo["disc"], DISC_R)
        # restrict to pixels that are actually one of the lamp colours (safety)
        lamp = np.zeros((h, w), bool)
        for c in COLORS.values():
            lamp |= (base == np.array(c, dtype=base.dtype)).all(axis=2)
        masks[name] = m & lamp
    font = ImageFont.truetype(FONT, FONT_SIZE)

    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS),
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "medium",
            "-movflags", "+faststart", OUT,
        ],
        stdin=subprocess.PIPE,
    )
    for i in range(N_FRAMES):
        t = sim_time(i)
        if i == 0:
            frame = base  # exact first frame
        else:
            frame = render_frame(base, masks, font, t)
        ffmpeg.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
    ffmpeg.stdin.close()
    ffmpeg.wait()
    if ffmpeg.returncode != 0:
        raise SystemExit("ffmpeg failed")

    # Also save first/last frames for inspection
    Image.fromarray(render_frame(base, masks, font, 0.0)).save(os.path.join(OUT_DIR, "frame_first_rendered.png"))
    Image.fromarray(render_frame(base, masks, font, float(SIM_SECONDS))).save(os.path.join(OUT_DIR, "frame_last.png"))
    for name in LIGHTS:
        print(name, "t=0:", state_at(name, 0.0), " t=6:", state_at(name, 6.0))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
