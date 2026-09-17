#!/usr/bin/env python3
"""Generate the mirror-reflection video from first_frame.png.

Action: the angle annotation (arc + "theta = 55 deg" text) fades out while the
reflected ray (reflectivity 0.97) grows from the incidence point to the right
edge of the image, obeying the law of reflection.
"""
import subprocess, os
import numpy as np
from PIL import Image, ImageDraw

APP = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(APP, "first_frame.png")
OUT = os.path.join(APP, "output", "video.mp4")
N_FRAMES, FPS, W, H = 35, 16, 1024, 1024

base = np.array(Image.open(SRC).convert("RGB")).astype(np.float32)

# --- geometry measured from the first frame -------------------------------
# incident ray centre line: passes (200, 243.5) and (300, 312.5)  -> slope 0.69
slope = (312.5 - 243.5) / 100.0
def ray_y(x): return 243.5 + slope * (x - 200)
MIRROR_TOP, MIRROR_BOT = 416, 421           # black mirror band rows
NORMAL_X = 453                              # grey normal line column
P = (453.0, ray_y(453))                     # incidence point (~418)
REFLECTIVITY = 0.97
BLUE = np.array([0, 0, 255], np.float32)
WHITE = np.array([255, 255, 255], np.float32)
REF_COLOR = tuple(int(round(c)) for c in REFLECTIVITY * BLUE + (1 - REFLECTIVITY) * WHITE)
RAY_W = 4

# --- build the "clean" plate: annotation removed, ray gap repaired --------
clean = base.copy()
above = np.zeros((H, W), bool); above[:MIRROR_TOP] = True
grey = (base[..., 0] == base[..., 1]) & (base[..., 1] == base[..., 2]) & (base[..., 0] < 255)
normal = np.zeros((H, W), bool); normal[:, NORMAL_X] = True
annot = above & grey & ~(normal & (base[..., 0] == 150))
clean[annot] = WHITE
# repair incident-ray pixels that the arc had been drawn over
yy, xx = np.mgrid[0:H, 0:W]
band = np.abs(yy - ray_y(xx)) <= RAY_W / 2
clean[annot & band & (xx < NORMAL_X)] = BLUE

# --- reflected ray ----------------------------------------------------------
d_in = np.array([1.0, slope]); d_in /= np.linalg.norm(d_in)
d_ref = np.array([d_in[0], -d_in[1]])       # reflect about the (vertical) normal
t_edge = (W - 1 - P[0]) / d_ref[0]           # reach right image edge
E = (P[0] + t_edge * d_ref[0], P[1] + t_edge * d_ref[1])

def draw_reflected(img_arr, frac):
    if frac <= 0: return img_arr
    im = Image.fromarray(img_arr.astype(np.uint8))
    dr = ImageDraw.Draw(im)
    tip = (P[0] + frac * t_edge * d_ref[0], P[1] + frac * t_edge * d_ref[1])
    dr.line([P, tip], fill=REF_COLOR, width=RAY_W)
    # open arrowhead (two barbs at +/-30 deg, ~30 px) like the incident ray's
    L = min(30.0, frac * t_edge)
    back = -d_ref
    for a in (np.radians(30), -np.radians(30)):
        c, s = np.cos(a), np.sin(a)
        v = (back[0] * c - back[1] * s, back[0] * s + back[1] * c)
        dr.line([tip, (tip[0] + L * v[0], tip[1] + L * v[1])], fill=REF_COLOR, width=3)
    return np.array(im).astype(np.float32)

# --- frames -----------------------------------------------------------------
FADE_FRAMES = 8
frames = []
for i in range(N_FRAMES):
    if i == 0:
        fr = base.copy()
    else:
        a = min(1.0, i / FADE_FRAMES)             # annotation fade-out
        fr = base * (1 - a) + clean * a
        frac = (i / (N_FRAMES - 1))               # ray growth, linear pacing
        fr = draw_reflected(fr, frac)
    frames.append(np.clip(np.round(fr), 0, 255).astype(np.uint8))

raw = b"".join(f.tobytes() for f in frames)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT],
               input=raw, check=True)
Image.fromarray(frames[-1]).save(os.path.join(APP, "output", "last_frame.png"))
print("wrote", OUT, "end point", E, "color", REF_COLOR)
