#!/usr/bin/env python3
"""Animate colored animal faces moving (straight line) onto their matching outlines."""
import itertools
import subprocess
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS, N_FRAMES = 16, 64

img = np.array(Image.open(SRC).convert("RGB"))
H, W, _ = img.shape
bg = np.array([245, 250, 255], dtype=np.uint8)
fg = np.any(img != bg, axis=2)

# Divider: find the vertical line column(s) spanning almost the full height
col_counts = fg.sum(axis=0)
div_cols = np.where(col_counts > 0.9 * H)[0]
div_x0, div_x1 = div_cols.min(), div_cols.max()

def components(mask, dilate=6):
    """Group nearby foreground pixels into objects; return list of exact masks."""
    grown = ndi.binary_dilation(mask, iterations=dilate)
    lab, n = ndi.label(grown)
    objs = []
    for i in range(1, n + 1):
        m = mask & (lab == i)
        if m.sum() < 50:
            continue
        objs.append(m)
    return objs

left_mask = fg.copy(); left_mask[:, div_x0 - 2:] = False
right_mask = fg.copy(); right_mask[:, :div_x1 + 3] = False
faces = components(left_mask)
outlines = components(right_mask)
assert len(faces) == len(outlines), (len(faces), len(outlines))

def silhouette(m):
    return ndi.binary_fill_holes(m)

def bbox_center(m):
    ys, xs = np.where(m)
    return (ys.min() + ys.max()) / 2.0, (xs.min() + xs.max()) / 2.0

def shift_to(m, cy, cx):
    y0, x0 = bbox_center(m)
    return ndi.shift(m.astype(np.uint8), (round(cy - y0), round(cx - x0)), order=0).astype(bool)

face_sil = [silhouette(m) for m in faces]
out_sil = [silhouette(m) for m in outlines]

# Similarity matrix: IoU of silhouettes aligned on bbox centers
n = len(faces)
S = np.zeros((n, n))
for i, fs in enumerate(face_sil):
    for j, os_ in enumerate(out_sil):
        a = shift_to(fs, H / 2, W / 2)
        b = shift_to(os_, H / 2, W / 2)
        S[i, j] = (a & b).sum() / max(1, (a | b).sum())
best = max(itertools.permutations(range(n)), key=lambda p: sum(S[i, p[i]] for i in range(n)))
print("similarity matrix:\n", np.round(S, 2))
print("assignment:", best)

# Background = frame with faces erased
base = img.copy()
allf = np.zeros_like(fg)
for m in faces:
    allf |= m
base[allf] = bg

# Sprites
sprites = []
for i, m in enumerate(faces):
    ys, xs = np.where(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    rgba = np.zeros((y1 - y0, x1 - x0, 4), dtype=np.uint8)
    rgba[..., :3] = img[y0:y1, x0:x1]
    rgba[..., 3] = m[y0:y1, x0:x1] * 255
    sc = bbox_center(m)
    tc = bbox_center(outlines[best[i]])
    sprites.append((rgba, (y0, x0), np.array(sc), np.array(tc)))

def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)

frames = []
for k in range(N_FRAMES):
    t = k / (N_FRAMES - 1)
    e = ease(t)
    frame = Image.fromarray(base.copy())
    for rgba, (y0, x0), sc, tc in sprites:
        d = (tc - sc) * e
        sp = Image.fromarray(rgba)
        frame.paste(sp, (int(round(x0 + d[1])), int(round(y0 + d[0]))), sp)
    frames.append(np.array(frame))

frames[0] = img.copy()  # exact first frame

ff = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
     "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
     "-crf", "16", "-preset", "medium", OUT], stdin=subprocess.PIPE)
for f in frames:
    ff.stdin.write(np.ascontiguousarray(f).tobytes())
ff.stdin.close(); ff.wait()
Image.fromarray(frames[-1]).save("/app/output/last_frame.png")
print("wrote", OUT)
