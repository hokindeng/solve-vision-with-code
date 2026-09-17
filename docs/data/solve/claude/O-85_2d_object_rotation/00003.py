#!/usr/bin/env python3
"""Rotate the single 2D object in first_frame.png clockwise by 180 degrees
around its centroid over 17 frames, keeping everything else unchanged."""
import os
import subprocess
import numpy as np
from PIL import Image
from scipy import ndimage

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "first_frame.png")
OUT_DIR = os.path.join(BASE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES = 17
FPS = 16
TOTAL_DEG = 180.0  # clockwise


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    img = Image.open(SRC).convert("RGB")
    arr = np.array(img)
    H, W, _ = arr.shape

    # Non-background pixels (background is pure white).
    nonwhite = np.any(arr < 250, axis=2)
    # Exclude the title text at the top: find connected components, pick the
    # largest one (the object); the title glyphs are many small components.
    lab, n = ndimage.label(nonwhite)
    sizes = ndimage.sum(nonwhite, lab, range(1, n + 1))
    obj_label = int(np.argmax(sizes)) + 1
    mask = lab == obj_label
    # Fill interior holes and grow slightly to catch anti-aliased edge pixels.
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_dilation(mask, iterations=2)

    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()

    # Background with the object erased (uniform white behind it).
    bg = arr.copy()
    bg[mask] = 255

    # Object layer as RGBA.
    alpha = (mask * 255).astype(np.uint8)
    obj_rgba = np.dstack([arr, alpha])
    obj_img = Image.fromarray(obj_rgba, "RGBA")

    frames = []
    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)
        angle_cw = TOTAL_DEG * t
        if i == 0:
            frame = Image.fromarray(arr)
        else:
            # PIL rotates counter-clockwise for positive angles.
            rot = obj_img.rotate(-angle_cw, resample=Image.BICUBIC,
                                 center=(cx, cy), expand=False)
            frame = Image.fromarray(bg).convert("RGBA")
            frame.alpha_composite(rot)
            frame = frame.convert("RGB")
        frames.append(frame)

    tmp_dir = os.path.join(OUT_DIR, "_frames")
    os.makedirs(tmp_dir, exist_ok=True)
    for i, f in enumerate(frames):
        f.save(os.path.join(tmp_dir, f"f{i:03d}.png"))

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp_dir, "f%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
        "-r", str(FPS), OUT,
    ], check=True)
    for fn in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, fn))
    os.rmdir(tmp_dir)
    print(f"wrote {OUT}: {len(frames)} frames, centroid=({cx:.1f},{cy:.1f})")


if __name__ == "__main__":
    main()
