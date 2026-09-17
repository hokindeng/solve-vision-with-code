#!/usr/bin/env python3
"""Generate the mirror-reflection video from first_frame.png.

Action: fade out the angle annotation (arc + "theta = 68deg" label), then grow the
reflected ray from the hit point to the image edge.  The mirror has reflectivity
0.38, so the reflected ray is drawn with 38% intensity of the incident ray colour
(blue blended over the white background).
"""
import os
import subprocess
import numpy as np
import cv2
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

W = H = 1024
FPS = 16
N_FRAMES = 35

REFLECTIVITY = 0.38

# Geometry measured from first_frame.png
HIT = np.array([563.0, 610.0])          # where the incident ray meets the mirror
INC_SLOPE = 0.41358605                  # dy/dx of incident ray (going down-right)
RAY_COLOR = np.array([0, 0, 255])       # incident ray colour (RGB)
ARROW_LEN, ARROW_HW = 14, 9             # arrowhead length / half width (px)

# Angle annotation region (arc + text) and the normal line passing through it
ANN_Y0, ANN_Y1 = 568, 606
ANN_X0, ANN_X1 = 523, 697
NORMAL_X, NORMAL_Y0 = 563, 580
NORMAL_COLOR = (150, 150, 150)


def smoothstep(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def build_clean(base):
    """Background with the angle annotation removed (everything else unchanged)."""
    clean = base.copy()
    reg = clean[ANN_Y0:ANN_Y1, ANN_X0:ANN_X1]
    # keep the incident ray (pure blue) which passes through the annotation box;
    # erase only the black/grey arc + text pixels
    is_ray = (reg[..., 2] > 200) & (reg[..., 0] < 80) & (reg[..., 1] < 80)
    reg[~is_ray] = 255
    # restore the portion of the normal line that ran through the annotation box
    clean[NORMAL_Y0:ANN_Y1, NORMAL_X] = NORMAL_COLOR
    return clean


def reflected_dir():
    ang = np.arctan(INC_SLOPE)
    return np.array([np.cos(ang), -np.sin(ang)])   # up-right


def ray_end_at_edge():
    d = reflected_dir()
    # reach x = W-1 first (the ray is shallow), compute y there
    t = (W - 1 - HIT[0]) / d[0]
    return HIT + t * d


def draw_reflected(img, frac):
    """Draw the reflected ray from HIT covering `frac` of its full length."""
    if frac <= 0:
        return img
    d = reflected_dir()
    n = np.array([-d[1], d[0]])
    end_full = ray_end_at_edge()
    tip = HIT + (end_full - HIT) * frac

    layer = np.zeros((H, W), np.uint8)
    cv2.line(layer, tuple(np.round(HIT).astype(int)), tuple(np.round(tip).astype(int)),
             255, 2, cv2.LINE_8)
    length = np.linalg.norm(tip - HIT)
    L = min(ARROW_LEN, length)
    hw = ARROW_HW * (L / ARROW_LEN)
    base = tip - L * d
    tri = np.array([tip, base + hw * n, base - hw * n]).round().astype(np.int32)
    cv2.fillPoly(layer, [tri], 255)

    mask = layer > 0
    # Only draw above the mirror surface (never over the mirror / hatching)
    mask[int(HIT[1]) - 1:, :] &= False
    out = img.copy()
    src = out[mask].astype(np.float64)
    out[mask] = np.round(src * (1 - REFLECTIVITY) + RAY_COLOR * REFLECTIVITY).astype(np.uint8)
    return out


def make_frame(base, clean, i):
    # Phase 1 (frames 1..9): fade annotation away.  Phase 2 (frames 7..34): grow ray.
    fade = smoothstep((i - 0) / 9.0)
    grow = smoothstep((i - 7) / (N_FRAMES - 1 - 7))
    if i == 0:
        frame = base.copy()
    else:
        frame = base.copy()
        reg = (slice(ANN_Y0, ANN_Y1), slice(ANN_X0, ANN_X1))
        frame[reg] = np.round(base[reg] * (1 - fade) + clean[reg] * fade).astype(np.uint8)
    return draw_reflected(frame, grow)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = np.array(Image.open(FIRST).convert("RGB"))
    clean = build_clean(base)

    frames_dir = os.path.join(OUT_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for i in range(N_FRAMES):
        fr = make_frame(base, clean, i)
        Image.fromarray(fr).save(os.path.join(frames_dir, f"{i:04d}.png"))

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
           "-i", os.path.join(frames_dir, "%04d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "10", "-preset", "slow",
           "-r", str(FPS), OUT]
    subprocess.run(cmd, check=True)
    import shutil
    shutil.rmtree(frames_dir, ignore_errors=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
