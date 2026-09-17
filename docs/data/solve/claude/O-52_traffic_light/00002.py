#!/usr/bin/env python3
"""Regenerate /app/output/video.mp4: four independent traffic lights counting down
for 4 simulated seconds, starting from /app/first_frame.png.

Only the light disks (fill colour) and the countdown digits are redrawn; every
other pixel is copied unchanged from the first frame.
"""
import math
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(ROOT, "first_frame.png")
OUT_DIR = os.path.join(ROOT, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

FPS = 16
N_FRAMES = 80
SIM_SECONDS = 4.0
PHASE_LEN = 4.0

FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 102)

COLORS = {"red": (255, 0, 0), "yellow": (255, 200, 0), "green": (0, 200, 0)}
# 3-colour cycle as listed in the prompt: R -> Y -> G -> Y -> R
CYCLE = ["red", "yellow", "green", "yellow"]

# Geometry measured from first_frame.png.
# circle: centre + radius (outline 3 px, no anti-aliasing); box: interior bbox (inclusive).
LIGHTS = {
    "N": dict(circle=(512, 214, 65), box=(451, 573, 276, 398), phase=0, remaining=3.0),
    # South is yellow; taken as the yellow that follows red (first yellow in the cycle).
    "S": dict(circle=(512, 798, 65), box=(451, 573, 860, 982), phase=1, remaining=4.0),
    "E": dict(circle=(804, 506, 65), box=(743, 865, 568, 690), phase=2, remaining=1.0),
    "W": dict(circle=(220, 506, 65), box=(159, 281, 568, 690), phase=2, remaining=1.0),
}


def state_at(light, t):
    """Return (colour name, displayed countdown) of a light after t simulated seconds."""
    phase = light["phase"]
    rem = light["remaining"] - t
    eps = 1e-9
    while rem <= eps:  # countdown reached 0 -> switch to next phase
        phase = (phase + 1) % len(CYCLE)
        rem += PHASE_LEN
    return CYCLE[phase], int(math.ceil(rem - eps))


def build_masks(base):
    """Per-light boolean mask of the disk fill pixels (exact fill colour inside circle)."""
    masks = {}
    for name, l in LIGHTS.items():
        cx, cy, r = l["circle"]
        inside = np.zeros(base.shape[:2], dtype=bool)
        inside[cy - r:cy + r + 1, cx - r:cx + r + 1] = True  # circle bounding box
        fill = np.array(COLORS[CYCLE[l["phase"]]])
        is_fill = np.all(base == fill, axis=2)
        masks[name] = inside & is_fill
    return masks


def render(base, masks, t):
    frame = base.copy()
    for name, l in LIGHTS.items():
        color, count = state_at(l, t)
        frame[masks[name]] = COLORS[color]
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    for name, l in LIGHTS.items():
        color, count = state_at(l, t)
        x0, x1, y0, y1 = l["box"]
        d.rectangle([x0, y0, x1, y1], fill=(255, 255, 255))
        # placement reproduces the original digits pixel-exactly
        d.text((x0 + 60.5, y0 + 63.5), str(count), fill=(0, 0, 0), font=FONT, anchor="mm")
    return np.array(img)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = np.array(Image.open(FIRST).convert("RGB"))
    masks = build_masks(base)

    frames = []
    for i in range(N_FRAMES):
        t = SIM_SECONDS * i / (N_FRAMES - 1)  # t=0 on the first frame, t=4 s on the last
        frames.append(render(base, masks, t))

    # sanity: the first frame must reproduce the source image exactly
    assert np.array_equal(frames[0], base), "first frame does not match first_frame.png"

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1024x1024", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "medium",
        "-movflags", "+faststart", OUT,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print("wrote", OUT, "frames:", len(frames))
    for name, l in LIGHTS.items():
        print(name, "final:", state_at(l, SIM_SECONDS))


if __name__ == "__main__":
    main()
