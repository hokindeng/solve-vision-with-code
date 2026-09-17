#!/usr/bin/env python3
"""Majority-color task: count objects per color, keep the majority color,
fade every other object (fill + outline) to the background.

Usage: python3 /app/solve.py  ->  /app/output/video.mp4
"""
import os
import subprocess
from collections import Counter

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT = os.path.join(HERE, "output", "video.mp4")
N_FRAMES, FPS = 40, 16


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def main():
    img = np.array(Image.open(SRC).convert("RGB"))
    h, w, _ = img.shape

    # Background = most common color (white here). Outline = black.
    flat = img.reshape(-1, 3)
    bg = np.array(Counter(map(tuple, flat)).most_common(1)[0][0], np.uint8)
    nonbg = np.any(img != bg, axis=2)
    outline = np.all(img == 0, axis=2)
    fill = nonbg & ~outline

    # Connected fill components -> one object each; record its color.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fill.astype(np.uint8), connectivity=8)
    obj_color = {}
    for lab in range(1, n):
        if stats[lab, cv2.CC_STAT_AREA] < 30:  # ignore antialias specks
            continue
        ys, xs = np.nonzero(labels == lab)
        cols = Counter(map(tuple, img[ys, xs]))
        obj_color[lab] = cols.most_common(1)[0][0]

    counts = Counter(obj_color.values())
    majority = counts.most_common(1)[0][0]
    print("counts per color:", dict(counts), "-> majority:", majority)

    # Assign outline (and any unlabeled non-bg) pixels to the nearest labeled object.
    lab_img = np.where(np.isin(labels, list(obj_color)), labels, 0).astype(np.int32)
    kernel = np.ones((3, 3), np.uint8)
    while True:
        todo = nonbg & (lab_img == 0)
        if not todo.any():
            break
        grown = cv2.dilate(lab_img.astype(np.float32), kernel).astype(np.int32)
        new = todo & (grown > 0)
        if not new.any():
            break
        lab_img[new] = grown[new]

    vanish_labels = [lab for lab, c in obj_color.items() if c != majority]
    print("objects vanishing:", len(vanish_labels), "kept:", len(obj_color) - len(vanish_labels))

    # Stagger fades slightly so the action spans the whole clip.
    rng = np.random.default_rng(0)
    order = sorted(vanish_labels, key=lambda l: stats[l, cv2.CC_STAT_TOP] + stats[l, cv2.CC_STAT_LEFT])
    starts = np.linspace(0.0, 0.35, len(order)) if len(order) > 1 else np.array([0.0])
    fade_len = 0.65  # in units of total duration; last fade ends exactly at t=1

    bg_img = np.empty_like(img)
    bg_img[:] = bg

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-sws_flags", "area+accurate_rnd+full_chroma_int",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-qp", "0", "-preset", "slow", OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)
        alpha = np.zeros((h, w), np.float32)  # 0 = original, 1 = background
        for lab, s in zip(order, starts):
            a = smoothstep((t - s) / fade_len)
            if a > 0:
                alpha[lab_img == lab] = a
        if i == N_FRAMES - 1:
            alpha[np.isin(lab_img, order)] = 1.0
        frame = (img.astype(np.float32) * (1 - alpha[..., None]) + bg_img * alpha[..., None])
        frame = np.clip(np.rint(frame), 0, 255).astype(np.uint8)
        if i == 0:
            assert np.array_equal(frame, img)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    assert proc.returncode == 0, "ffmpeg failed"
    print("wrote", OUT)


if __name__ == "__main__":
    main()
