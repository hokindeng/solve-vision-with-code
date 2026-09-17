#!/usr/bin/env python3
"""Majority-color task: count colored objects, keep the majority color, fade the rest out.

Frame 0 is exactly first_frame.png; the non-majority objects (fill + their own black
outline) blend smoothly into the background over the clip; everything else is untouched.
"""
import os
import subprocess

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 40


def main():
    img = np.array(Image.open(SRC).convert("RGB"))
    h, w, _ = img.shape

    # Background = most frequent color; outline = black.
    flat = img.reshape(-1, 3)
    cols, counts = np.unique(flat, axis=0, return_counts=True)
    bg = cols[np.argmax(counts)]
    bg_mask = np.all(img == bg, axis=2)
    outline_mask = np.all(img < 40, axis=2)

    # Object fill colors = every remaining color with a meaningful pixel count.
    fill_colors = [tuple(c) for c, n in zip(cols, counts)
                   if not np.array_equal(c, bg) and not np.all(c < 40) and n > 50]

    # Label every fill pixel with a component id; remember color per component.
    comp_lab = np.zeros((h, w), np.int32)
    comp_color = [None]  # index 0 = none
    per_color_count = {}
    for c in fill_colors:
        m = np.all(img == np.array(c, np.uint8), axis=2).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        k = 0
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] < 30:
                continue
            comp_lab[lab == i] = len(comp_color)
            comp_color.append(c)
            k += 1
        per_color_count[c] = k
    print("object counts per color:", per_color_count)
    majority = max(per_color_count, key=per_color_count.get)
    print("majority color:", majority)

    # Assign each outline pixel to its nearest fill component.
    fill_any = comp_lab > 0
    _, idx = cv2.distanceTransformWithLabels(
        (~fill_any).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(fill_any)
    pix_to_comp = np.zeros(idx.max() + 1, np.int32)
    pix_to_comp[idx[ys, xs]] = comp_lab[ys, xs]
    owner = pix_to_comp[idx]  # component owning each pixel by proximity
    owner[fill_any] = comp_lab[fill_any]

    # Pixels that vanish: fills of non-majority objects + outline pixels owned by them.
    is_minor_comp = np.array([False] + [c != majority for c in comp_color[1:]])
    vanish = is_minor_comp[owner] & (fill_any | outline_mask)
    # Never touch pixels that belong to a majority fill or the background.
    vanish &= ~bg_mask
    print("vanishing pixels:", int(vanish.sum()))

    # Target image after vanishing: background where minor objects were, except where a
    # majority object was partly hidden behind a vanishing one -- there we reconstruct the
    # occluded part by fitting a rectangle / ellipse to the visible pixels.
    target = img.copy()
    target[vanish] = bg
    change = vanish.copy()  # every pixel that is allowed to differ from the source
    k3 = np.ones((3, 3), np.uint8)
    thick = 2  # outline thickness in px (measured from the source drawing)
    for cid in range(1, len(comp_color)):
        if is_minor_comp[cid]:
            continue
        obj = (owner == cid) & (fill_any | outline_mask)
        if not (cv2.dilate(obj.astype(np.uint8), k3).astype(bool) & vanish).any():
            continue
        ys_, xs_ = np.nonzero(obj)
        x0, x1, y0, y1 = xs_.min(), xs_.max(), ys_.min(), ys_.max()
        rect = np.zeros((h, w), np.uint8)
        rect[y0:y1 + 1, x0:x1 + 1] = 1
        ell = np.zeros((h, w), np.uint8)
        cv2.ellipse(ell, (int(x0 + x1) // 2, int(y0 + y1) // 2), (int(x1 - x0) // 2, int(y1 - y0) // 2), 0, 0, 360, 1, -1)
        best, best_name, best_score = None, None, -1
        roi = np.zeros((h, w), bool)
        roi[max(0, y0 - 2):y1 + 3, max(0, x0 - 2):x1 + 3] = True
        sel = roi & ~vanish  # pixels whose true content we can see
        for name, cand in (("rect", rect), ("ellipse", ell)):
            c = cand.astype(bool)
            score = ((c == obj) & sel).sum() / sel.sum()
            if score > best_score:
                best, best_name, best_score = c, name, score
        inner = cv2.erode(best.astype(np.uint8), np.ones((2 * thick + 1, 2 * thick + 1), np.uint8)).astype(bool)
        recon_outline = best & ~inner & vanish
        # Interior of the reconstructed shape: vanished pixels plus any stray outline pixels
        # left behind by the occluding object's border.
        recon_fill = inner & (vanish | outline_mask)
        target[recon_outline] = 0
        target[recon_fill] = np.array(comp_color[cid], np.uint8)
        change |= recon_fill
        print(f"reconstructed occluded object {cid}: fit={best_name}, score={best_score:.3f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    base = img.astype(np.float32)
    tgt = target.astype(np.float32)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-r", str(FPS), OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    fade_end = N_FRAMES - 4  # fully vanished, then hold a few frames
    for f in range(N_FRAMES):
        t = min(1.0, f / fade_end)
        a = 0.5 - 0.5 * np.cos(np.pi * t)  # smooth ease in/out
        frame = base.copy()
        frame[change] = base[change] * (1 - a) + tgt[change] * a
        proc.stdin.write(np.clip(frame + 0.5, 0, 255).astype(np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
