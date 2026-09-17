#!/usr/bin/env python3
"""Move each colored animal face to its matching dark outline (straight line,
all simultaneous, eased), keeping every other pixel unchanged."""
import os, subprocess, shutil, tempfile
import numpy as np
import cv2
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT = os.path.join(HERE, "output", "video.mp4")
N_FRAMES, FPS = 64, 16

base = np.array(Image.open(SRC).convert("RGB"))
H, W, _ = base.shape
bg = base[5, 5].copy()

# --- segment everything that is not background -----------------------------
mask = np.abs(base.astype(int) - bg.astype(int)).sum(-1) > 0
lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
comps = []
for i in range(1, n + 1):
    ys, xs = np.where(lab == i)
    comps.append(dict(id=i, x0=xs.min(), x1=xs.max(), y0=ys.min(), y1=ys.max(),
                      n=len(ys), col=base[ys, xs].mean(0)))

def is_dark(c):
    return c["col"].max() < 120

# faces: colourful components on the left half
faces = [c for c in comps if c["x1"] < W // 2 and not is_dark(c) and c["n"] > 500]
# outlines: dark components on the right half (exclude the thin divider line)
outl = [c for c in comps if c["x0"] > W // 2 and is_dark(c)]

# merge outline pieces whose bboxes overlap/are close (e.g. fox stripes + head)
def merge(groups):
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a, b = groups[i], groups[j]
                if (a["x0"] <= b["x1"] + 4 and b["x0"] <= a["x1"] + 4 and
                        a["y0"] <= b["y1"] + 4 and b["y0"] <= a["y1"] + 4):
                    groups[i] = dict(x0=min(a["x0"], b["x0"]), x1=max(a["x1"], b["x1"]),
                                     y0=min(a["y0"], b["y0"]), y1=max(a["y1"], b["y1"]))
                    groups.pop(j)
                    changed = True
                    break
            if changed:
                break
    return groups
outl = merge([dict(x0=c["x0"], x1=c["x1"], y0=c["y0"], y1=c["y1"]) for c in outl])
assert len(faces) == len(outl) == 4, (len(faces), len(outl))

def size(c):
    return (c["x1"] - c["x0"] + 1, c["y1"] - c["y0"] + 1)
def center(c):
    return ((c["x0"] + c["x1"]) / 2.0, (c["y0"] + c["y1"]) / 2.0)

# --- match faces to outlines by bounding-box size (unique optimal assignment)
from itertools import permutations
best = None
for perm in permutations(range(4)):
    cost = sum(abs(size(faces[i])[0] - size(outl[perm[i]])[0]) +
               abs(size(faces[i])[1] - size(outl[perm[i]])[1]) for i in range(4))
    if best is None or cost < best[0]:
        best = (cost, perm)
perm = best[1]

# --- build sprites and a clean background with faces erased -----------------
clean = base.copy()
sprites = []
for i, f in enumerate(faces):
    m = (lab == f["id"])
    ys, xs = np.where(m)
    x0, x1, y0, y1 = f["x0"], f["x1"], f["y0"], f["y1"]
    rgba = np.zeros((y1 - y0 + 1, x1 - x0 + 1, 4), np.uint8)
    rgba[..., :3] = base[y0:y1 + 1, x0:x1 + 1]
    rgba[..., 3] = (m[y0:y1 + 1, x0:x1 + 1] * 255).astype(np.uint8)
    clean[m] = bg
    fc, oc = center(f), center(outl[perm[i]])
    sprites.append(dict(img=rgba, x0=x0, y0=y0, dx=oc[0] - fc[0], dy=oc[1] - fc[1]))

def ease(t):  # smooth ease-in-out
    return t * t * (3 - 2 * t)

def render(t):
    frame = clean.astype(np.float32)
    for s in sprites:
        tx, ty = s["dx"] * t, s["dy"] * t
        M = np.float32([[1, 0, s["x0"] + tx], [0, 1, s["y0"] + ty]])
        warped = cv2.warpAffine(s["img"], M, (W, H), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        a = warped[..., 3:4].astype(np.float32) / 255.0
        frame = frame * (1 - a) + warped[..., :3].astype(np.float32) * a
    return np.clip(frame + 0.5, 0, 255).astype(np.uint8)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
tmp = tempfile.mkdtemp()
for k in range(N_FRAMES):
    t = ease(k / (N_FRAMES - 1))
    img = base if k == 0 else render(t)
    Image.fromarray(img).save(os.path.join(tmp, f"{k:04d}.png"))
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                "-i", os.path.join(tmp, "%04d.png"), "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-crf", "12", "-r", str(FPS), OUT], check=True)
shutil.rmtree(tmp)
print("wrote", OUT)
