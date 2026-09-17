#!/usr/bin/env python3
"""Majority-color task: count objects per color, keep the majority color,
fade every other-colored object (fill + its outline) into the background."""
import os
import subprocess
import numpy as np
import cv2
from PIL import Image

SRC = "/app/first_frame.png"
OUT_DIR = "/app/output"
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS = 16
N_FRAMES = 40
BG = np.array([255, 255, 255], np.uint8)
OUTLINE = np.array([0, 0, 0], np.uint8)


def main():
    img = np.array(Image.open(SRC).convert("RGB"))
    h, w, _ = img.shape

    is_bg = np.all(img == BG, axis=2)
    is_outline = np.all(img == OUTLINE, axis=2)
    fill = ~is_bg & ~is_outline

    # Count objects per fill color (connected components of same-colored fill).
    colors = np.unique(img[fill].reshape(-1, 3), axis=0)
    counts = {}
    masks = {}
    for c in colors:
        m = np.all(img == c, axis=2).astype(np.uint8)
        n, lab = cv2.connectedComponents(m, connectivity=4)
        # ignore specks (n includes background label 0)
        sizes = [(lab == i).sum() for i in range(1, n)]
        objs = sum(1 for s in sizes if s >= 30)
        counts[tuple(int(x) for x in c)] = objs
        masks[tuple(int(x) for x in c)] = m.astype(bool)
    majority = max(counts, key=counts.get)
    print("object counts per color:", counts)
    print("majority color:", majority)

    keep_fill = masks[majority]
    remove_fill = fill & ~keep_fill

    # Outline pixels belong to the nearest fill; outline shared with a kept
    # object stays so kept shapes remain fully intact.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    near_remove = cv2.dilate(remove_fill.astype(np.uint8), k).astype(bool)
    near_keep = cv2.dilate(keep_fill.astype(np.uint8), k).astype(bool)
    remove_outline = is_outline & near_remove & ~near_keep
    remove = remove_fill | remove_outline

    # Final frame: removed pixels become background.
    final = img.copy()
    final[remove] = BG

    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = os.path.join(OUT_DIR, "_frames")
    os.makedirs(tmp, exist_ok=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))

    # Fade out smoothly over the whole clip (frame 0 == first_frame exactly).
    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)
        a = 0.5 - 0.5 * np.cos(np.pi * t)  # ease in/out, 0 -> 1
        frame = img.astype(np.float32)
        frame[remove] = (1 - a) * img[remove].astype(np.float32) + a * BG.astype(np.float32)
        frame = np.clip(np.round(frame), 0, 255).astype(np.uint8)
        if i == 0:
            frame = img.copy()
        if i == N_FRAMES - 1:
            frame = final.copy()
        Image.fromarray(frame).save(os.path.join(tmp, f"f{i:03d}.png"))

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp, "f%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
        "-r", str(FPS), OUT,
    ], check=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
