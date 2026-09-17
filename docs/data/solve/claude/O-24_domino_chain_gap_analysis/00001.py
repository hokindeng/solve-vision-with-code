"""Domino chain: push domino 1; 1 and 2 fall, chain stops at the wide gap before 3."""
import math, subprocess, os
import numpy as np
from PIL import Image

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
N_FRAMES, FPS = 50, 16
GROUND_Y = 724                 # first ground-line row (dominos stand on it)
GROUND_RGB = (118, 85, 43)
DOM_TOP = 583
# x extents (inclusive) of the two dominos that fall
DOMS = [(151, 196), (225, 270)]

base = Image.open(SRC).convert("RGB")

# --- extract sprites (exact pixels) and erase them from the background ------
sprites = []
bg = base.copy()
for x0, x1 in DOMS:
    spr = base.crop((x0, DOM_TOP, x1 + 1, GROUND_Y + 1)).convert("RGBA")
    sprites.append(spr)
    px = bg.load()
    for y in range(DOM_TOP, GROUND_Y + 1):
        for x in range(x0, x1 + 1):
            px[x, y] = (255, 255, 255) if y < GROUND_Y - 1 else GROUND_RGB
W_D = DOMS[0][1] - DOMS[0][0] + 1        # 46
H_D = GROUND_Y - DOM_TOP + 1             # 142

# --- physics-ish angles ----------------------------------------------------
gap12 = DOMS[1][0] - DOMS[0][1] - 1                  # 28 px between 1 and 2
th_contact = math.degrees(math.asin(gap12 / (H_D - 1)))   # 1 touches 2
# final rest of 1: right edge touches top-left corner of flat domino 2
dx = DOMS[1][1] - DOMS[0][1]                          # pivot-to-pivot
th_rest1 = math.degrees(math.atan2(dx, W_D))          # ~58 deg
th_rest2 = 90.0

def ease_in(t):   return t * t
def ease_out(t):  return 1 - (1 - t) ** 2

def angles(f):
    """Return (theta1, theta2) in degrees, clockwise tilt, for frame f."""
    t_hold, t_c, t_end = 4, 18, 42
    if f <= t_hold:
        return 0.0, 0.0
    if f <= t_c:
        u = (f - t_hold) / (t_c - t_hold)
        return th_contact * ease_in(u), 0.0
    if f <= t_end:
        u = (f - t_c) / (t_end - t_c)
        # simple accelerate-then-settle curve
        s = u * u * (3 - 2 * u)  # smoothstep
        s = s ** 1.15
        return th_contact + (th_rest1 - th_contact) * s, th_rest2 * s
    return th_rest1, th_rest2

def draw_domino(canvas, spr, x0, x1, theta):
    """Rotate sprite clockwise by theta about its bottom-right corner (x1, GROUND_Y)."""
    SS = 4
    big = spr.resize((spr.width * SS, spr.height * SS), Image.LANCZOS)
    rot = big.rotate(-theta, resample=Image.BICUBIC, expand=True)
    # position of pivot (bottom-right corner) after rotation, relative to rot image
    cx, cy = big.width / 2, big.height / 2
    vx, vy = big.width - cx, big.height - cy            # pivot rel. to centre
    a = math.radians(-theta)                            # PIL rotates CCW for +angle; our -theta
    # PIL rotate: image coords, rotating by angle counter-clockwise (visual)
    rx = vx * math.cos(a) + vy * math.sin(a)
    ry = -vx * math.sin(a) + vy * math.cos(a)
    px = rot.width / 2 + rx
    py = rot.height / 2 + ry
    small = rot.resize((max(1, round(rot.width / SS)), max(1, round(rot.height / SS))), Image.LANCZOS)
    ox = x1 + 1 - px / SS
    oy = GROUND_Y + 1 - py / SS
    canvas.alpha_composite(small, (int(round(ox)), int(round(oy))))

def render(f):
    th1, th2 = angles(f)
    canvas = bg.convert("RGBA")
    # draw 2 first, then 1 (1 leans on 2 at the end)
    draw_domino(canvas, sprites[1], *DOMS[1], th2)
    draw_domino(canvas, sprites[0], *DOMS[0], th1)
    return canvas.convert("RGB")

os.makedirs("/app/output", exist_ok=True)
frames_dir = "/app/output/frames"
os.makedirs(frames_dir, exist_ok=True)
for f in range(N_FRAMES):
    img = base if f == 0 else render(f)
    img.save(f"{frames_dir}/{f:03d}.png")

subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                "-i", f"{frames_dir}/%03d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "16", OUT], check=True)
print("wrote", OUT)
