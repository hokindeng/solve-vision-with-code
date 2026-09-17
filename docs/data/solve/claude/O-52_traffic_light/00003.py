#!/usr/bin/env python3
"""Regenerate /app/output/video.mp4: simulate three traffic lights for 6 s.

Cycle per light: Red(4s) -> Yellow(4s) -> Green(4s) -> Yellow(4s) -> Red ...
Initial state (t=0): West red 4s left, North yellow 1s left, East green 1s left.
The video is 7.0 s at 16 fps (112 frames): 6 s of real-time simulation followed
by a 1 s hold on the final state. Only the light discs and the countdown digits
are redrawn; every other pixel is copied from first_frame.png.
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
SIM_SECONDS = 6.0
PHASE_LEN = 4.0

COLORS = {"red": (255, 0, 0), "yellow": (255, 200, 0), "green": (0, 200, 0)}
# Phase sequence (index cycles mod 4).
CYCLE = ["red", "yellow", "green", "yellow"]

# Geometry measured from first_frame.png:
# light: circle centre, radius (fill region); box: white countdown box (x0,x1,y0,y1)
LIGHTS = {
    "west":  {"center": (220, 506), "box": (159, 281, 568, 690), "phase": 0, "remaining": 4.0},
    "north": {"center": (512, 214), "box": (451, 573, 276, 398), "phase": 1, "remaining": 1.0},
    "east":  {"center": (804, 506), "box": (743, 865, 568, 690), "phase": 2, "remaining": 1.0},
}
# North is "yellow" -> the yellow following red (index 1), so it turns green next.


def state_at(light, t):
    """Return (color_name, seconds_remaining_display) for a light at time t."""
    phase = light["phase"]
    remaining = light["remaining"]
    # advance through phases
    while t >= remaining - 1e-9:
        t -= remaining
        phase = (phase + 1) % 4
        remaining = PHASE_LEN
    left = remaining - t
    display = int(math.ceil(left - 1e-6))
    display = max(1, min(display, int(PHASE_LEN)))
    return CYCLE[phase], display


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = np.array(Image.open(FIRST).convert("RGB"))
    font = ImageFont.truetype(FONT, 102)

    # Fill masks for each light: pixels currently equal to that light's fill colour
    # inside the disc's bounding box (fills are flat, un-antialiased).
    masks = {}
    for name, L in LIGHTS.items():
        cx, cy = L["center"]
        col = np.array(COLORS[CYCLE[L["phase"]]], dtype=np.uint8)
        sub = np.zeros(base.shape[:2], dtype=bool)
        y0, y1, x0, x1 = cy - 70, cy + 70, cx - 70, cx + 70
        sub[y0:y1, x0:x1] = np.all(base[y0:y1, x0:x1] == col, axis=2)
        masks[name] = sub

    def render(t):
        frame = base.copy()
        img = None
        for name, L in LIGHTS.items():
            color, digits = state_at(L, t)
            frame[masks[name]] = COLORS[color]
            # redraw countdown box interior (white) + digit
            x0, x1, y0, y1 = L["box"]
            frame[y0 + 1:y1, x0 + 1:x1] = (255, 255, 255)
            if img is None:
                img = Image.fromarray(frame)
                draw = ImageDraw.Draw(img)
            else:
                img = Image.fromarray(frame)
                draw = ImageDraw.Draw(img)
            txt = str(digits)
            bb = draw.textbbox((0, 0), txt, font=font)
            w, h = bb[2] - bb[0], bb[3] - bb[1]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            draw.text((cx - w / 2 - bb[0], cy - h / 2 - bb[1]), txt, font=font, fill=(0, 0, 0))
            frame = np.array(img)
        return frame

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1024x1024", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "medium",
        "-r", str(FPS), OUT,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(N_FRAMES):
        t = min(SIM_SECONDS, i / FPS)
        frame = render(t)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print("wrote", OUT)
    for name, L in LIGHTS.items():
        print(name, "final:", state_at(L, SIM_SECONDS))


if __name__ == "__main__":
    main()
