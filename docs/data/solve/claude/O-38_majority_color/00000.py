#!/usr/bin/env python3
"""Majority-color task: count objects per color, keep the majority color,
fade every other-colored object (fill + outline) to background."""
import os
import subprocess
import numpy as np
import cv2
from PIL import Image

SRC = "/app/first_frame.png"
OUT_DIR = "/app/output"
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES = 40
FPS = 16


def main():
    base = np.array(Image.open(SRC).convert("RGB")).astype(np.uint8)
    h, w, _ = base.shape

    # Background = the dominant color of the image (white here).
    cols, counts = np.unique(base.reshape(-1, 3), axis=0, return_counts=True)
    order = np.argsort(-counts)
    bg = cols[order[0]]
    non_bg = np.any(base != bg, axis=2)

    # Fill colors: saturated (non-gray) colors. Outlines are near-black/gray.
    def is_gray(c):
        return int(c.max()) - int(c.min()) < 40

    fill_colors = [c for c, n in zip(cols, counts)
                   if not np.array_equal(c, bg) and not is_gray(c) and n > 50]

    # Count objects per color via connected components of each fill mask.
    color_masks, color_counts = {}, {}
    for c in fill_colors:
        m = np.all(base == c, axis=2).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        # ignore specks smaller than 30 px (anti-alias noise)
        objs = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 30]
        color_masks[tuple(c)] = m.astype(bool)
        color_counts[tuple(c)] = len(objs)
        print(f"color {tuple(int(v) for v in c)}: {len(objs)} objects")

    majority = max(color_counts, key=color_counts.get)
    print("majority color:", majority)

    # Assign every non-background pixel (incl. outline) to nearest fill color.
    dists = []
    keys = list(color_masks.keys())
    for k in keys:
        inv = (~color_masks[k]).astype(np.uint8)
        dists.append(cv2.distanceTransform(inv, cv2.DIST_L2, 5))
    nearest = np.argmin(np.stack(dists, 0), axis=0)
    owner_is_majority = np.array([k == majority for k in keys])[nearest]
    vanish = non_bg & ~owner_is_majority          # pixels to remove
    print("pixels to vanish:", int(vanish.sum()))

    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = os.path.join(OUT_DIR, "frames")
    os.makedirs(tmp, exist_ok=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))

    bgf = bg.astype(np.float32)
    basef = base.astype(np.float32)
    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)                    # 0 .. 1
        # smooth ease-in-out fade; frame 0 identical to source, last = gone
        a = 0.5 - 0.5 * np.cos(np.pi * t)
        frame = basef.copy()
        frame[vanish] = (1 - a) * basef[vanish] + a * bgf
        if i == N_FRAMES - 1:
            frame[vanish] = bgf
        frame = np.clip(np.round(frame), 0, 255).astype(np.uint8)
        Image.fromarray(frame).save(os.path.join(tmp, f"{i:04d}.png"))

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp, "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
        "-preset", "slow", OUT], check=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
