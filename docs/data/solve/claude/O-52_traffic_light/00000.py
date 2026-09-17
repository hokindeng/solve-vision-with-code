#!/usr/bin/env python3
"""Regenerate /app/output/video.mp4: four traffic lights simulated for 6 s.

Cycle per light: Red(4s) -> Yellow(4s) -> Green(4s) -> Yellow(4s) -> Red ...
Initial state (all yellow, taken as the yellow that follows red):
  North 1s, South 2s, East 4s, West 1s remaining.
Only the light discs and the countdown digits are redrawn; every other pixel
is copied from first_frame.png.
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

FPS = 16
N_FRAMES = 112               # 7.0 s
HOLD_START = 0.5             # seconds of the initial state before action
SIM_SECONDS = 6.0            # simulated time covered by the video

FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 102)

YELLOW = (255, 200, 0)
COLORS = {"red": (220, 0, 0), "yellow": YELLOW, "green": (0, 200, 0)}
PHASES = ["red", "yellow", "green", "yellow"]
PHASE_LEN = 4.0

# name: (disc bbox x0,y0,x1,y1 inclusive incl. outline), (box interior x0,y0,x1,y1 inclusive),
#       initial phase index, initial seconds remaining
LIGHTS = {
    "N": ((440, 145, 585, 285), (451, 276, 573, 398), 1, 1.0),
    "S": ((440, 730, 585, 870), (451, 860, 573, 982), 1, 2.0),
    "E": ((730, 440, 880, 575), (743, 568, 865, 690), 1, 4.0),
    "W": ((145, 440, 295, 575), (159, 568, 281, 690), 1, 1.0),
}


def state_at(phase_idx, remaining, tau):
    """Advance a light by tau seconds; return (color, seconds remaining)."""
    while tau >= remaining:
        tau -= remaining
        phase_idx = (phase_idx + 1) % len(PHASES)
        remaining = PHASE_LEN
    return PHASES[phase_idx], remaining - tau


def render(base, masks, tau):
    img = base.copy()
    arr = np.array(img)
    for name, (disc, box, idx, rem) in LIGHTS.items():
        color, remaining = state_at(idx, rem, tau)
        x0, y0, x1, y1 = disc
        sub = arr[y0:y1 + 1, x0:x1 + 1]
        sub[masks[name]] = COLORS[color]
    img = Image.fromarray(arr)
    dr = ImageDraw.Draw(img)
    for name, (disc, box, idx, rem) in LIGHTS.items():
        color, remaining = state_at(idx, rem, tau)
        digit = str(int(math.ceil(remaining - 1e-9)))
        bx0, by0, bx1, by1 = box
        dr.rectangle([bx0, by0, bx1, by1], fill=(255, 255, 255))
        cx = (bx0 + bx1 + 1) / 2.0
        cy = (by0 + by1 + 1) / 2.0
        bb = dr.textbbox((0, 0), digit, font=FONT)
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        dr.text((cx - w / 2 - bb[0], cy - h / 2 - bb[1]), digit, fill=(0, 0, 0), font=FONT)
    return img


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = Image.open(FIRST).convert("RGB")
    barr = np.array(base)
    masks = {}
    for name, (disc, _, _, _) in LIGHTS.items():
        x0, y0, x1, y1 = disc
        sub = barr[y0:y1 + 1, x0:x1 + 1]
        masks[name] = np.all(sub == np.array(YELLOW, dtype=np.uint8), axis=2)

    frames = []
    for i in range(N_FRAMES):
        t = i / FPS
        tau = min(max(t - HOLD_START, 0.0), SIM_SECONDS)
        frames.append(render(base, masks, tau))

    # sanity: frame 0 must equal the reference exactly
    assert np.array_equal(np.array(frames[0]), barr), "frame 0 differs from first_frame.png"

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1024x1024", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-movflags", "+faststart", OUT,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.array(f).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    frames[-1].save(os.path.join(OUT_DIR, "last_frame.png"))
    print("wrote", OUT, len(frames), "frames")


if __name__ == "__main__":
    main()
