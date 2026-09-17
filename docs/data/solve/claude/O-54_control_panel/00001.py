#!/usr/bin/env python3
"""Generate the control-panel video: move levers of units 1 and 3 from the
right position to the middle position so all indicator lights turn blue.

Inferred rule from first_frame.png: middle lever -> blue light,
right lever -> purple light.  Unit 2 is already correct.
"""
import os
import subprocess
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

FPS = 16
N_FRAMES = 24

# ---- geometry measured from first_frame.png -------------------------------
PANEL_X0 = [87, 394, 701]          # left edge of each panel (incl. 2px border)
PANEL_Y0, PANEL_Y1 = 618, 735      # inclusive
BORDER = 2
KNOB_SIZE = 57
KNOB_Y0 = 648
KNOB_LEFT_OFFSET = {"left": 7, "middle": 89, "right": 166}   # knob x0 - panel x0
DOT_CENTERS = [35, 117, 199]       # dot center x offsets from panel x0
DOT_Y = 676                        # dot center row
LIGHT_BOX = [(157, 202, 252, 297), (465, 202, 560, 297), (772, 202, 867, 297)]

BLACK = np.array([0, 0, 0], np.uint8)
GRAY = np.array([128, 128, 128], np.uint8)
DOT = np.array([240, 240, 240], np.uint8)
PURPLE = np.array([128, 0, 128], np.uint8)
BLUE = np.array([0, 0, 255], np.uint8)

# Units whose lever must change: (unit index, from, to)
MOVES = [(0, "right", "middle"), (2, "right", "middle")]


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * t)


def draw_panel(frame, unit, knob_x0):
    """Redraw one panel's interior with the knob at absolute x0 = knob_x0."""
    px0 = PANEL_X0[unit]
    x0, x1 = px0 + BORDER, px0 + 236 - BORDER   # interior [x0, x1)
    y0, y1 = PANEL_Y0 + BORDER, PANEL_Y1 + 1 - BORDER
    frame[y0:y1, x0:x1] = BLACK
    # three position dots (5x5 with corners clipped)
    for cx in DOT_CENTERS:
        ax = px0 + cx
        frame[DOT_Y - 2:DOT_Y + 3, ax - 2:ax + 3] = DOT
        for dy, dx in ((-2, -2), (-2, 2), (2, -2), (2, 2)):
            frame[DOT_Y + dy, ax + dx] = BLACK
    # knob
    kx = int(round(knob_x0))
    frame[KNOB_Y0:KNOB_Y0 + KNOB_SIZE, kx:kx + KNOB_SIZE] = GRAY


def set_light(frame, base, unit, color):
    x0, y0, x1, y1 = LIGHT_BOX[unit]
    region = base[y0:y1 + 1, x0:x1 + 1]
    mask = np.all(region == PURPLE, axis=2)
    out = frame[y0:y1 + 1, x0:x1 + 1]
    out[mask] = color


def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    os.makedirs(OUT_DIR, exist_ok=True)

    # Sequential schedule: each move takes a slice of the timeline.
    n_moves = len(MOVES)
    span = (N_FRAMES - 2) / n_moves        # frames per move (frame 0 static, last static)
    frames = []
    for i in range(N_FRAMES):
        f = base.copy()
        for m, (unit, src, dst) in enumerate(MOVES):
            start = 1 + m * span
            t = (i - start) / (span - 1)
            t = ease(t)
            px0 = PANEL_X0[unit]
            kx = px0 + KNOB_LEFT_OFFSET[src] + t * (KNOB_LEFT_OFFSET[dst] - KNOB_LEFT_OFFSET[src])
            if i > 0:
                draw_panel(f, unit, kx)
            if t >= 1.0:
                set_light(f, base, unit, BLUE)
        frames.append(f)

    # sanity: first frame identical to the source image
    assert np.array_equal(frames[0], base)

    tmp = os.path.join(OUT_DIR, "_frames")
    os.makedirs(tmp, exist_ok=True)
    for i, f in enumerate(frames):
        Image.fromarray(f).save(os.path.join(tmp, f"{i:04d}.png"))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp, "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT,
    ], check=True)
    for name in os.listdir(tmp):
        os.remove(os.path.join(tmp, name))
    os.rmdir(tmp)
    print("wrote", OUT, "frames:", len(frames))


if __name__ == "__main__":
    main()
