#!/usr/bin/env python3
"""Move each object horizontally to the right into its dashed target outline."""
import os, subprocess, tempfile
import numpy as np
from PIL import Image
from scipy import ndimage

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
N_FRAMES, FPS = 30, 16

img = np.array(Image.open(SRC).convert("RGB")).astype(np.float64)
H, W, _ = img.shape
nonwhite = (img < 250).any(2)

# --- segment objects: large connected non-white components ---
lab, n = ndimage.label(nonwhite)
sizes = ndimage.sum(nonwhite, lab, range(1, n + 1))
obj_ids = [i + 1 for i, s in enumerate(sizes) if s > 1000]
obj_masks = [lab == i for i in obj_ids]
allobj = np.zeros((H, W), bool)
for m in obj_masks:
    allobj |= m

# dashed-outline pixels = non-white pixels not belonging to objects
dashed = nonwhite & ~ndimage.binary_dilation(allobj, iterations=2)

def alpha_and_color(mask):
    """Estimate per-pixel alpha and pure color of an anti-aliased object on white."""
    pix = img[mask]
    # palette: dominant exact colors within the object (fill + outline)
    cols, cnt = np.unique(pix.astype(np.uint8), axis=0, return_counts=True)
    pal = cols[cnt > 0.01 * cnt.sum()].astype(np.float64)
    pal = pal[(pal < 250).any(1)]
    best_err = np.full(len(pix), np.inf)
    alpha = np.ones(len(pix))
    color = np.zeros_like(pix)
    for C in pal:
        denom = 255.0 - C
        ch = np.argmax(denom)
        a = np.clip((255.0 - pix[:, ch]) / denom[ch], 0, 1)
        recon = a[:, None] * C + (1 - a[:, None]) * 255.0
        err = np.abs(recon - pix).sum(1)
        sel = err < best_err
        best_err[sel] = err[sel]; alpha[sel] = a[sel]; color[sel] = C
    return alpha, color

objects = []
for m in obj_masks:
    ys, xs = np.where(m)
    a, c = alpha_and_color(m)
    # boundary of the object (for matching against dashed outline)
    edge = m & ~ndimage.binary_erosion(m, iterations=2)
    eys, exs = np.where(ndimage.binary_dilation(edge, iterations=2))
    # find horizontal shift maximising overlap of boundary with dashed pixels
    best, bestdx = -1, 0
    for dx in range(0, W - exs.max()):
        nx = exs + dx
        score = dashed[eys, nx].sum()
        if score > best:
            best, bestdx = score, dx
    objects.append(dict(ys=ys, xs=xs, alpha=a, color=c, dx=bestdx))
    print(f"object bbox x[{xs.min()},{xs.max()}] y[{ys.min()},{ys.max()}] -> dx={bestdx} (score {best})")

# background with objects removed (they sit on pure white)
bg = img.copy()
bg[allobj] = 255.0

def ease(t):
    return t * t * (3 - 2 * t)  # smoothstep

frames = []
for i in range(N_FRAMES):
    t = ease(i / (N_FRAMES - 1))
    fr = bg.copy()
    for o in objects:
        dx = int(round(o["dx"] * t))
        nx = o["xs"] + dx
        a = o["alpha"][:, None]
        fr[o["ys"], nx] = a * o["color"] + (1 - a) * fr[o["ys"], nx]
    if i == 0:
        fr = img.copy()  # first frame identical to source
    frames.append(np.clip(fr, 0, 255).astype(np.uint8))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with tempfile.TemporaryDirectory() as td:
    for i, f in enumerate(frames):
        Image.fromarray(f).save(f"{td}/f{i:03d}.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", f"{td}/f%03d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "12", "-preset", "slow", OUT], check=True)
    Image.fromarray(frames[-1]).save("/app/output/last_frame.png")
print("wrote", OUT)
