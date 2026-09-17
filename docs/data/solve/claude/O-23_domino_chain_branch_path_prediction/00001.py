#!/usr/bin/env python3
"""Generate the domino chain-reaction video from first_frame.png."""
import math, subprocess, os
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
W = H = 1024
FPS = 16
N_FRAMES = 62

frame0 = Image.open(SRC).convert("RGB")
arr0 = np.array(frame0)
BG = tuple(int(v) for v in arr0[5, 5])

# Domino bodies measured from the frame: (name, x0, x1, y0, y1, crop_x0, crop_x1)
# crop range widens for the START label, which overflows its body.
DOMINOS = [
    ("START", 62, 101, 450, 552, 46, 118),
    ("T1", 187, 225, 450, 552, 187, 225),
    ("T2", 311, 349, 450, 552, 311, 349),
    ("T3", 435, 473, 450, 552, 435, 473),
    ("A1", 559, 597, 357, 459, 559, 597),
    ("B1", 559, 597, 543, 645, 559, 597),
    ("A2", 683, 721, 336, 438, 683, 721),
    ("A3", 807, 845, 315, 417, 807, 845),
    ("A4", 931, 969, 294, 396, 931, 969),
    ("B2", 683, 721, 563, 665, 683, 721),
]
X_MARK = (624, 656, 588, 620)

# Fall schedule: (start_frame, duration_frames, final_angle_deg)
LEAN = 62.0    # rests against the next domino
FLAT = 90.0    # nothing to stop it
STEP, DUR = 6, 12
SCHED = {
    "START": (2 + 0 * STEP, DUR, LEAN),
    "T1": (2 + 1 * STEP, DUR, LEAN),
    "T2": (2 + 2 * STEP, DUR, LEAN),
    "T3": (2 + 3 * STEP, DUR, LEAN),
    "A1": (2 + 4 * STEP, DUR, LEAN),
    "B1": (2 + 4 * STEP, DUR + 2, FLAT),   # falls into the gap
    "A2": (2 + 5 * STEP, DUR, LEAN),
    "A3": (2 + 6 * STEP, DUR, LEAN),
    "A4": (2 + 7 * STEP, DUR, 42.0),      # end of branch A; stays inside the frame
    "B2": None,                            # never falls
}


def make_sprite(x0, x1, y0, y1, cx0, cx1):
    """Full-canvas RGBA layer holding just this domino (and its label)."""
    layer = np.zeros((H, W, 4), np.uint8)
    reg = arr0[y0:y1 + 1, cx0:cx1 + 1]
    diff = np.abs(reg.astype(int) - np.array(BG)).sum(2) > 0
    alpha = np.zeros(diff.shape, np.uint8)
    alpha[diff] = 255
    alpha[:, x0 - cx0:x1 - cx0 + 1] = 255  # whole body opaque
    layer[y0:y1 + 1, cx0:cx1 + 1, :3] = reg
    layer[y0:y1 + 1, cx0:cx1 + 1, 3] = alpha
    return Image.fromarray(layer, "RGBA")


sprites = {}
base = arr0.copy()
for name, x0, x1, y0, y1, cx0, cx1 in DOMINOS:
    sprites[name] = make_sprite(x0, x1, y0, y1, cx0, cx1)
    reg = base[y0:y1 + 1, cx0:cx1 + 1]
    reg[:] = BG  # erase domino from the background plate
base_img = Image.fromarray(base, "RGB")

# The red X stays on top of everything.
xl = np.zeros((H, W, 4), np.uint8)
x0, x1, y0, y1 = X_MARK
reg = arr0[y0:y1 + 1, x0:x1 + 1]
m = np.abs(reg.astype(int) - np.array(BG)).sum(2) > 0
xl[y0:y1 + 1, x0:x1 + 1, :3] = reg
xl[y0:y1 + 1, x0:x1 + 1, 3] = m.astype(np.uint8) * 255
x_layer = Image.fromarray(xl, "RGBA")


def angle_at(name, f):
    s = SCHED[name]
    if s is None or f <= s[0]:
        return 0.0
    start, dur, final = s
    t = min(1.0, (f - start) / dur)
    ease = t ** 1.9  # gravity-like acceleration
    if final >= 89 and t >= 0.85:      # small settle bounce for flat falls
        ease = 1.0 - 0.06 * math.sin((t - 0.85) / 0.15 * math.pi)
    return final * ease


def render(f):
    img = base_img.convert("RGBA")
    layers = []
    for name, x0, x1, y0, y1, cx0, cx1 in DOMINOS:
        a = angle_at(name, f)
        spr = sprites[name]
        if a > 0:
            spr = spr.rotate(-a, resample=Image.BICUBIC, center=(x1 + 0.5, y1 + 1.0))
        layers.append((a, spr))
    # fallen ones first, standing ones on top (B2 must cover the flat B1)
    for a, spr in sorted(layers, key=lambda t: -t[0]):
        img.alpha_composite(spr)
    img.alpha_composite(x_layer)
    return img.convert("RGB")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "medium",
           "-r", str(FPS), OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(N_FRAMES):
        fr = frame0 if f == 0 else render(f)
        p.stdin.write(np.asarray(fr, np.uint8).tobytes())
    p.stdin.close()
    p.wait()
    assert p.returncode == 0
    print("wrote", OUT)


if __name__ == "__main__":
    main()
