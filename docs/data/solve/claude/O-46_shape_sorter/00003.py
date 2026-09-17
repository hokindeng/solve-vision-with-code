#!/usr/bin/env python3
"""Shape sorter: slide each colored card from the left staging area into its
matching outline on the right. Everything except the moving card is kept
pixel-identical to first_frame.png."""
import os, subprocess, shutil
import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 80

first = np.array(Image.open(SRC).convert("RGB")).astype(np.float32)
H, W, _ = first.shape
bg = first[5, 5].copy()

# --- segment all non-background components -------------------------------
mask = np.abs(first - bg).sum(2) > 10
lab, n = ndimage.label(mask)
comps = []
for i in range(1, n + 1):
    ys, xs = np.where(lab == i)
    px = first[ys, xs].astype(int)
    vals, cnt = np.unique(px.reshape(-1, 3), axis=0, return_counts=True)
    dom = vals[cnt.argmax()]
    comps.append(dict(label=i, x0=xs.min(), x1=xs.max(), y0=ys.min(), y1=ys.max(),
                      n=len(xs), dom=dom.astype(np.float32),
                      cx=(xs.min() + xs.max()) / 2, cy=(ys.min() + ys.max()) / 2))

OUTLINE_COL = np.array([100, 116, 139], np.float32)
cards = [c for c in comps if c["x1"] < 500 and np.abs(c["dom"] - OUTLINE_COL).sum() > 30]
outlines = [c for c in comps if c["x0"] > 520 and np.abs(c["dom"] - OUTLINE_COL).sum() < 30]

# Order required by the prompt: yellow triangle, blue diamond, cyan star, orange circle
def card_kind(c):
    d = c["dom"]
    if d[0] > 200 and d[1] > 180 and d[2] < 80: return "triangle"   # yellow
    if d[1] > 180 and d[2] > 200 and d[0] < 100: return "star"       # cyan
    if d[2] > 200 and d[0] < 150: return "diamond"                   # blue
    return "circle"                                                  # orange

# outlines: match by shape geometry via row/column position (same layout as cards)
def kind_from_layout(c):
    top = c["cy"] < 512
    left = c["cx"] < (min(o["cx"] for o in outlines) + max(o["cx"] for o in outlines)) / 2
    return {(True, True): "triangle", (True, False): "diamond",
            (False, True): "star", (False, False): "circle"}[(top, left)]

outline_by_kind = {kind_from_layout(o): o for o in outlines}
card_by_kind = {card_kind(c): c for c in cards}
ORDER = ["triangle", "diamond", "star", "circle"]

# --- build sprites and a clean background (cards erased) -----------------
background = first.copy()
sprites = {}
PAD = 2
for k in ORDER:
    c = card_by_kind[k]
    x0, x1, y0, y1 = c["x0"] - PAD, c["x1"] + PAD, c["y0"] - PAD, c["y1"] + PAD
    region = first[y0:y1 + 1, x0:x1 + 1]
    dvec = c["dom"] - bg
    # alpha = projection of (pixel - bg) onto (dom - bg)
    alpha = ((region - bg) * dvec).sum(2) / max((dvec ** 2).sum(), 1e-6)
    alpha = np.clip(alpha, 0, 1)
    # only keep this component's pixels (avoid grabbing neighbors)
    comp_mask = ndimage.binary_dilation(lab[y0:y1 + 1, x0:x1 + 1] == c["label"], iterations=2)
    alpha *= comp_mask
    sprites[k] = dict(alpha=alpha, color=c["dom"], ox=c["cx"] - x0, oy=c["cy"] - y0)
    background[y0:y1 + 1, x0:x1 + 1] = np.where(comp_mask[..., None], bg, background[y0:y1 + 1, x0:x1 + 1])

def paste(img, spr, cx, cy):
    a = spr["alpha"]
    h, w = a.shape
    x0 = int(round(cx - spr["ox"])); y0 = int(round(cy - spr["oy"]))
    sub = img[y0:y0 + h, x0:x0 + w]
    img[y0:y0 + h, x0:x0 + w] = sub * (1 - a[..., None]) + spr["color"] * a[..., None]

def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)

# --- animation schedule --------------------------------------------------
SEG = N_FRAMES // len(ORDER)      # 20 frames per card
MOVE_START, MOVE_END = 1, SEG - 2  # frames within a segment when the card slides
ARC = 120                          # peak vertical lift (px) of the slide path

frames = []
for f in range(N_FRAMES):
    img = background.copy()
    layers = []  # (z, sprite, cx, cy); moving card drawn last
    for i, k in enumerate(ORDER):
        c, o = card_by_kind[k], outline_by_kind[k]
        s0, s1 = i * SEG, (i + 1) * SEG
        if f < s0:
            t = 0.0
        elif f >= s1:
            t = 1.0
        else:
            t = np.clip((f - s0 - MOVE_START) / (MOVE_END - MOVE_START), 0, 1)
        moving = 0.0 < t < 1.0
        t = ease(t)
        cx = c["cx"] + (o["cx"] - c["cx"]) * t
        # gentle arc so the sliding card clears the other cards in its row
        cy = c["cy"] + (o["cy"] - c["cy"]) * t - ARC * np.sin(np.pi * t)
        layers.append((1 if moving else 0, sprites[k], cx, cy))
    for _, spr, cx, cy in sorted(layers, key=lambda L: L[0]):
        paste(img, spr, cx, cy)
    frames.append(np.clip(img + 0.5, 0, 255).astype(np.uint8))

# frame 0 must equal the source exactly
frames[0] = first.astype(np.uint8)

# --- encode --------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
tmp = os.path.join(OUT_DIR, "_frames")
shutil.rmtree(tmp, ignore_errors=True)
os.makedirs(tmp)
for i, fr in enumerate(frames):
    Image.fromarray(fr).save(os.path.join(tmp, f"{i:04d}.png"))
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                "-i", os.path.join(tmp, "%04d.png"), "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT], check=True)
shutil.rmtree(tmp, ignore_errors=True)
print("wrote", OUT, len(frames), "frames")
