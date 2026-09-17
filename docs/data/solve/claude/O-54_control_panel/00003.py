#!/usr/bin/env python3
"""Move the levers of units 1 and 3 to the right position (which the panel shows
yields indigo) and fade their indicator lights to indigo. Everything else stays
identical to first_frame.png."""
import subprocess, os
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 24, 16

base = np.array(Image.open(SRC).convert("RGB"))
H, W, _ = base.shape

# Measured geometry (all measured from first_frame.png)
PANEL_X0 = [87, 394, 701]          # panel left edge per unit
PANEL_Y0 = 618
INNER = (89, 620, 321, 734)        # x0,y0,x1,y1 inclusive of black interior (rel. to page, unit 1)
KNOB_OFF = {"left": 12, "middle": 89, "right": 173}   # knob x offset rel. to panel x0
KNOB_W, KNOB_Y0, KNOB_H = 57, 30, 57                   # knob y offset rel. to panel y0
DOT_OFF = [33, 115, 197]           # dot x offset (5 px wide), y offset 56..60
DOT_Y0, DOT_SZ = 56, 5
KNOB_COL = (128, 128, 128)
DOT_COL = (240, 240, 240)
BLACK = (0, 0, 0)

LIGHT_CX = [204, 512, 819]
LIGHT_CY = 249
LIGHT_R = 45.5
INDIGO = np.array([75, 0, 130], dtype=np.float64)

# Inferred relationship: left->purple, right->indigo, middle->pink. Target: all right.
current = ["left", "right", "middle"]
target = ["right", "right", "right"]
light_cols = [base[LIGHT_CY, cx].astype(np.float64) for cx in LIGHT_CX]

# Precompute light fill masks (pixels equal to the current fill colour, inside the circle)
yy, xx = np.mgrid[0:H, 0:W]
light_masks = []
for i, cx in enumerate(LIGHT_CX):
    inside = (xx - cx) ** 2 + (yy - LIGHT_CY) ** 2 <= (LIGHT_R + 2) ** 2
    same = np.all(base == light_cols[i].astype(np.uint8), axis=2)
    light_masks.append(inside & same)


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def draw_panel(img, unit, knob_x_off):
    x0 = PANEL_X0[unit]
    ix0, iy0, ix1, iy1 = INNER
    dx = x0 - PANEL_X0[0]
    img[iy0:iy1 + 1, ix0 + dx:ix1 + dx + 1] = BLACK
    for off in DOT_OFF:
        img[PANEL_Y0 + DOT_Y0:PANEL_Y0 + DOT_Y0 + DOT_SZ, x0 + off:x0 + off + DOT_SZ] = DOT_COL
    kx = x0 + int(round(knob_x_off))
    ky = PANEL_Y0 + KNOB_Y0
    img[ky:ky + KNOB_H, kx:kx + KNOB_W] = KNOB_COL


frames = []
MOVE_END = 16      # lever motion over frames 0..16
FADE_START, FADE_END = 12, 22  # light colour fades as the lever approaches/arrives
for f in range(N_FRAMES):
    img = base.copy()
    tm = ease(f / MOVE_END)
    tc = ease((f - FADE_START) / (FADE_END - FADE_START))
    for u in range(3):
        if current[u] == target[u]:
            continue
        a, b = KNOB_OFF[current[u]], KNOB_OFF[target[u]]
        draw_panel(img, u, a + (b - a) * tm)
        col = light_cols[u] * (1 - tc) + INDIGO * tc
        img[light_masks[u]] = np.round(col).astype(np.uint8)
    frames.append(img)

# Guarantee exact first frame and exact final state
frames[0] = base.copy()
assert np.array_equal(frames[0], base)

os.makedirs(OUT_DIR, exist_ok=True)
tmp = os.path.join(OUT_DIR, "_frames")
os.makedirs(tmp, exist_ok=True)
for i, fr in enumerate(frames):
    Image.fromarray(fr).save(os.path.join(tmp, f"{i:03d}.png"))
subprocess.run([
    "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
    "-i", os.path.join(tmp, "%03d.png"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
    "-r", str(FPS), OUT,
], check=True)
for fn in os.listdir(tmp):
    os.remove(os.path.join(tmp, fn))
os.rmdir(tmp)
print("wrote", OUT, len(frames), "frames")
