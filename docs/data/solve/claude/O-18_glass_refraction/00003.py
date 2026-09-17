#!/usr/bin/env python3
"""Generate the refraction video: erase the angle annotation, then draw the
red refracted ray (Snell's law) from the incidence point to the image boundary."""
import math
import subprocess
import numpy as np
from PIL import Image
import cv2

W = H = 1024
FPS = 16
N_FRAMES = 70
SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"

# Physical setup (from the prompt)
THETA_I = math.radians(41.1)
N_AIR, N_GLASS = 1.00, 1.627
THETA_R = math.asin(N_AIR * math.sin(THETA_I) / N_GLASS)

# Scene geometry measured from first_frame.png
P0 = (512, 512)            # incidence point on the interface
INTERFACE_BOTTOM = 513     # last row of the black interface line
RAY_THICK = 3              # blue ray thickness
RED = (255, 0, 0)
BLUE = (0, 0, 255)
NORMAL_GRAY = (150, 150, 150)
# bounding box of the arc + "θ = 41°" label (x0, y0, x1, y1), inclusive
ANNOT_BOX = (480, 468, 650, 510)

base = np.array(Image.open(SRC).convert("RGB"))

# ---- build the "annotation removed" background -------------------------
clean = base.copy()
x0, y0, x1, y1 = ANNOT_BOX
reg = clean[y0:y1 + 1, x0:x1 + 1]
is_blue = (reg == BLUE).all(axis=2)
xs = np.arange(x0, x1 + 1)[None, :].repeat(reg.shape[0], 0)
is_normal = (reg == NORMAL_GRAY).all(axis=2) & (xs == P0[0])
erase = ~is_blue & ~is_normal
reg[erase] = 255
annot_mask = np.zeros((H, W), bool)   # pixels that actually change when erasing
annot_mask[y0:y1 + 1, x0:x1 + 1] = erase & (base[y0:y1 + 1, x0:x1 + 1] != 255).any(axis=2)

# ---- refracted ray endpoint at the image boundary ----------------------
# incident ray travels down-right, so the refracted ray also goes down-right
dx, dy = math.sin(THETA_R), math.cos(THETA_R)
t = (H - 1 - P0[1]) / dy
tx = (W - 1 - P0[0]) / dx
t = min(t, tx)
P_END = (P0[0] + dx * t, P0[1] + dy * t)
RAY_LEN = t


def draw_ray(img, frac):
    """Draw the red ray from P0 for a fraction `frac` of its full length,
    only below the interface so the interface line stays untouched."""
    if frac <= 0:
        return img
    L = RAY_LEN * frac
    end = (P0[0] + dx * L, P0[1] + dy * L)
    layer = img.copy()
    cv2.line(layer, (int(round(P0[0])), int(round(P0[1]))),
             (int(round(end[0])), int(round(end[1]))), RED, RAY_THICK, cv2.LINE_8)
    out = img.copy()
    out[INTERFACE_BOTTOM + 1:] = layer[INTERFACE_BOTTOM + 1:]
    return out


def ease(u):
    return 0.5 - 0.5 * math.cos(math.pi * u)


# ---- timeline -----------------------------------------------------------
ERASE_START, ERASE_END = 4, 20     # annotation fades to white
RAY_START, RAY_END = 22, 66        # red ray grows to the boundary

frames = []
for i in range(N_FRAMES):
    if i < ERASE_START:
        img = base.copy()
    elif i < ERASE_END:
        a = ease((i - ERASE_START) / (ERASE_END - ERASE_START))
        img = base.copy()
        img[annot_mask] = np.clip(
            base[annot_mask] * (1 - a) + 255 * a, 0, 255).astype(np.uint8)
    else:
        img = clean.copy()
    if i >= RAY_START:
        frac = ease(min(1.0, (i - RAY_START) / (RAY_END - RAY_START)))
        img = draw_ray(img, frac)
    frames.append(img)

# first frame must be identical to the source
assert np.array_equal(frames[0], base)

Image.fromarray(frames[-1]).save("/app/output/last_frame.png")

cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
       "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
       "-c:v", "libx264", "-preset", "slow", "-crf", "10", "-pix_fmt", "yuv420p",
       "-movflags", "+faststart", OUT]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for f in frames:
    p.stdin.write(f.tobytes())
p.stdin.close()
p.wait()
if p.returncode != 0:
    raise SystemExit("ffmpeg failed")
print(f"theta_r = {math.degrees(THETA_R):.2f} deg, ray end = ({P_END[0]:.1f}, {P_END[1]:.1f})")
print("wrote", OUT)
