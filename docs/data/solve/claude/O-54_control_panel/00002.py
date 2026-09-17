#!/usr/bin/env python3
"""Move all three levers to the RIGHT position and turn the lights orange.

Inference from first_frame.png: middle lever -> purple, left lever -> green,
so the only remaining position (right) must be orange. Units 1 and 3 are in
the middle, unit 2 is on the left; all three move right.
"""
import subprocess
import numpy as np
from PIL import Image

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS, N = 16, 24

ORANGE = np.array([255, 165, 0], np.uint8)
KNOB = 128
DOT = 240
KNOB_SIZE = 57
KNOB_OFF = {"left": 11, "mid": 89, "right": 168}   # knob x0 relative to box x0
DOT_OFF = {"left": 33, "mid": 115, "right": 197}    # dot x0 relative to box x0
DOT_Y0 = 674
KNOB_Y0 = 648
BOX_Y0, BOX_Y1 = 618, 735  # outer (incl. 2px border)

# (box_x0, current lever position, light center x)
UNITS = [(87, "mid", 204), (394, "left", 511), (701, "mid", 818)]
LIGHT_CY, LIGHT_R = 249, 46

# dot shape: 5x5 with cut corners
DOT_MASK = np.array([[0, 1, 1, 1, 0],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [0, 1, 1, 1, 0]], bool)


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)


def draw_box(img, bx0, knob_x0):
    """Redraw one lever box interior with knob at knob_x0 (absolute)."""
    x0, x1 = bx0 + 2, bx0 + 236 - 2  # interior
    y0, y1 = BOX_Y0 + 2, BOX_Y1 - 1
    img[y0:y1, x0:x1] = 0
    for k in ("left", "mid", "right"):
        dx = bx0 + DOT_OFF[k]
        region = img[DOT_Y0:DOT_Y0 + 5, dx:dx + 5]
        region[DOT_MASK] = DOT
    img[KNOB_Y0:KNOB_Y0 + KNOB_SIZE, knob_x0:knob_x0 + KNOB_SIZE] = KNOB


def recolor_light(img, base, cx, alpha):
    """Blend the light's fill pixels toward orange by alpha (0..1)."""
    ys, xs = np.mgrid[LIGHT_CY - LIGHT_R:LIGHT_CY + LIGHT_R + 1,
                      cx - LIGHT_R:cx + LIGHT_R + 1]
    inside = (xs - cx) ** 2 + (ys - LIGHT_CY) ** 2 <= (LIGHT_R + 1) ** 2
    patch = base[LIGHT_CY - LIGHT_R:LIGHT_CY + LIGHT_R + 1,
                 cx - LIGHT_R:cx + LIGHT_R + 1]
    # fill pixels: saturated (non-gray) pixels, i.e. not black/white outline
    fill = inside & (patch.max(2).astype(int) - patch.min(2).astype(int) > 40)
    out = img[LIGHT_CY - LIGHT_R:LIGHT_CY + LIGHT_R + 1,
              cx - LIGHT_R:cx + LIGHT_R + 1]
    blended = (patch.astype(float) * (1 - alpha) + ORANGE.astype(float) * alpha)
    out[fill] = np.clip(blended[fill] + 0.5, 0, 255).astype(np.uint8)


def main():
    base = np.array(Image.open(SRC).convert("RGB"))
    frames = []
    move_start, move_end = 1, 16      # lever slides
    fade_start, fade_end = 14, 20     # light fades to orange (overlaps arrival)
    for i in range(N):
        img = base.copy()
        if i == 0:
            frames.append(img)
            continue
        t = np.clip((i - move_start) / (move_end - move_start), 0, 1)
        s = ease(t)
        a = ease(np.clip((i - fade_start) / (fade_end - fade_start), 0, 1))
        for bx0, pos, cx in UNITS:
            x_from = bx0 + KNOB_OFF[pos]
            x_to = bx0 + KNOB_OFF["right"]
            kx = int(round(x_from + (x_to - x_from) * s))
            draw_box(img, bx0, kx)
            if a > 0:
                recolor_light(img, base, cx, float(a))
        frames.append(img)

    # sanity: frame 0 identical to source
    assert np.array_equal(frames[0], base)

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", "1024x1024", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
         "-tune", "stillimage", OUT],
        stdin=subprocess.PIPE)
    for f in frames:
        ff.stdin.write(np.ascontiguousarray(f).tobytes())
    ff.stdin.close()
    ff.wait()
    Image.fromarray(frames[-1]).save("/app/output/last_frame.png")
    print("wrote", OUT, "frames:", len(frames))


if __name__ == "__main__":
    main()
