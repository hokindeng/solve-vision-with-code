#!/usr/bin/env python3
"""Shape sorter: slide each colored card from the left staging area into its matching outline."""
import os, subprocess, shutil, tempfile
import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(ROOT, "first_frame.png")
OUT = os.path.join(ROOT, "output", "video.mp4")
FPS, N_FRAMES = 16, 78

first = np.array(Image.open(FIRST).convert("RGB"))
H, W, _ = first.shape
bg_color = first[5, 5].copy()

# ---- segment every non-background component --------------------------------
mask = np.abs(first.astype(int) - bg_color.astype(int)).sum(2) > 30
lab, n = ndimage.label(mask)
comps = []
for i in range(1, n + 1):
    ys, xs = np.where(lab == i)
    if len(ys) < 50:
        continue
    col = first[ys[len(ys) // 2], xs[len(ys) // 2]]
    comps.append(dict(id=i, x0=xs.min(), x1=xs.max(), y0=ys.min(), y1=ys.max(), n=len(ys), col=col,
                      cx=(xs.min() + xs.max()) / 2, cy=(ys.min() + ys.max()) / 2))

OUTLINE_COL = np.array([100, 116, 139])
def is_outline(c): return np.abs(c["col"].astype(int) - OUTLINE_COL).sum() < 40
outlines = [c for c in comps if is_outline(c)]
cards = [c for c in comps if not is_outline(c) and c["x0"] < W // 2 and (c["x1"] - c["x0"]) < 200]

def near(c, rgb): return np.abs(c["col"].astype(int) - np.array(rgb)).sum() < 60
pink = next(c for c in cards if near(c, (244, 114, 182)))
red = next(c for c in cards if near(c, (248, 113, 113)))
yellow = next(c for c in cards if near(c, (250, 204, 21)))
outlines.sort(key=lambda c: c["cy"])  # hexagon (top), diamond (middle), star (bottom)
order = [(pink, outlines[0]), (red, outlines[1]), (yellow, outlines[2])]

# ---- background with cards removed; card sprites ----------------------------
background = first.copy()
sprites = []
for card, target in order:
    m = lab == card["id"]
    background[m] = bg_color
    ys, xs = np.where(m)
    sprite_mask = m[card["y0"]:card["y1"] + 1, card["x0"]:card["x1"] + 1]
    sprite_rgb = first[card["y0"]:card["y1"] + 1, card["x0"]:card["x1"] + 1]
    dx = int(round(target["cx"] - card["cx"]))
    dy = int(round(target["cy"] - card["cy"]))
    sprites.append(dict(rgb=sprite_rgb, mask=sprite_mask, x=card["x0"], y=card["y0"], dx=dx, dy=dy))

# ---- timeline ---------------------------------------------------------------
MOVE, GAP, HEAD = 22, 3, 1
starts = [HEAD + k * (MOVE + GAP) for k in range(3)]

def ease(t):  # smooth ease-in-out
    return 0.5 - 0.5 * np.cos(np.pi * t)

def progress(k, f):
    s = starts[k]
    if f < s: return 0.0
    if f >= s + MOVE: return 1.0
    return ease((f - s) / (MOVE - 1) if MOVE > 1 else 1.0)

def paste(canvas, sp, ox, oy):
    h, w = sp["mask"].shape
    region = canvas[oy:oy + h, ox:ox + w]
    region[sp["mask"]] = sp["rgb"][sp["mask"]]

outline_mask = np.zeros((H, W), bool)
for o in outlines:
    outline_mask |= lab == o["id"]

def render(f):
    canvas = background.copy()
    for k, sp in enumerate(sprites):
        p = progress(k, f)
        ox = int(round(sp["x"] + p * sp["dx"]))
        oy = int(round(sp["y"] + p * sp["dy"]))
        paste(canvas, sp, ox, oy)
    # outline strokes stay on top so they remain pixel-identical in every frame
    canvas[outline_mask] = first[outline_mask]
    return canvas

# ---- write frames & encode --------------------------------------------------
os.makedirs(os.path.dirname(OUT), exist_ok=True)
tmp = tempfile.mkdtemp(prefix="frames_")
for f in range(N_FRAMES):
    frame = render(f)
    if f == 0:
        assert np.array_equal(frame, first), "first frame must be identical to first_frame.png"
    Image.fromarray(frame).save(os.path.join(tmp, f"{f:04d}.png"))

subprocess.run([
    "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
    "-i", os.path.join(tmp, "%04d.png"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
    "-r", str(FPS), OUT
], check=True)
shutil.rmtree(tmp)
print("wrote", OUT)
