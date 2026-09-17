#!/usr/bin/env python3
"""Rotate the 3 objects in first_frame.png counterclockwise by 64 degrees
around their own centroids, producing /app/output/video.mp4."""
import os, subprocess, tempfile
import numpy as np
import cv2
from PIL import Image
from scipy import ndimage

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
N_FRAMES = 17
FPS = 16
TOTAL_DEG = 64.0
N_OBJECTS = 3

img = np.array(Image.open(SRC).convert("RGB"))
H, W, _ = img.shape

# Non-white pixels = ink (objects + title text). Antialiased edges included.
ink = np.any(img != 255, axis=2)
# Slightly dilate so antialiased fringes join their object, then label.
lab, n = ndimage.label(ndimage.binary_dilation(ink, iterations=2))
sizes = ndimage.sum(np.ones_like(lab), lab, index=range(1, n + 1))
# Skip components that are in the title band (small text glyphs near top).
comps = []
for i, s in enumerate(sizes, start=1):
    ys, xs = np.where(lab == i)
    if ys.min() < 70:  # title text region
        continue
    comps.append((s, i))
comps.sort(reverse=True)
objs = [i for _, i in comps[:N_OBJECTS]]

# Background: original with object regions erased to white.
background = img.copy()
layers = []
for i in objs:
    region = ndimage.binary_fill_holes(lab == i)
    alpha = (region & ink)  # actual object pixels
    # fill interior fully (fill color pixels are non-white anyway)
    alpha = ndimage.binary_fill_holes(alpha)
    ys, xs = np.where(alpha)
    cy, cx = ys.mean(), xs.mean()
    rgba = np.zeros((H, W, 4), np.uint8)
    rgba[..., :3] = img
    rgba[..., 3] = alpha.astype(np.uint8) * 255
    # A circle rotated about its centroid is pixel-identical; detect such
    # rotationally-symmetric objects and leave their pixels untouched to
    # avoid resampling noise on their edges.
    m8 = alpha.astype(np.uint8)
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = np.concatenate(cnts)
    (_, _), r = cv2.minEnclosingCircle(pts)
    fill_ratio = alpha.sum() / (np.pi * r * r)
    if fill_ratio > 0.9:
        continue  # stays in background as-is
    layers.append((rgba, (cx, cy)))
    background[region] = 255

def render(deg):
    frame = background.astype(np.float32)
    for rgba, (cx, cy) in layers:
        # positive angle in cv2 = counterclockwise (image y-down coordinates handled)
        M = cv2.getRotationMatrix2D((cx, cy), deg, 1.0)
        rot = cv2.warpAffine(rgba, M, (W, H), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        a = rot[..., 3:4].astype(np.float32) / 255.0
        frame = frame * (1 - a) + rot[..., :3].astype(np.float32) * a
    return np.clip(frame + 0.5, 0, 255).astype(np.uint8)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with tempfile.TemporaryDirectory() as td:
    for k in range(N_FRAMES):
        t = k / (N_FRAMES - 1)
        frame = img if k == 0 else render(TOTAL_DEG * t)
        Image.fromarray(frame).save(f"{td}/f{k:03d}.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", f"{td}/f%03d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "12", OUT], check=True)
print("wrote", OUT)
