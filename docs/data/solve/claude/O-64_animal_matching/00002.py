#!/usr/bin/env python3
"""Animate colored animal faces (left) moving onto their matching outlines (right).

Regenerate with:  python3 /app/solve.py
"""
import itertools
import os
import subprocess

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES = 64
FPS = 16
BG = np.array([245, 250, 255], dtype=np.uint8)
DIVIDER_X = 512  # vertical divider line lives at x=512..513


def components(mask, gap=12):
    """Group nearby non-background pixels into objects (ears etc. attach to faces)."""
    grown = ndimage.binary_dilation(mask, iterations=gap)
    labels, n = ndimage.label(grown)
    objs = []
    for i in range(1, n + 1):
        m = mask & (labels == i)
        if m.sum() < 200:
            continue
        ys, xs = np.nonzero(m)
        objs.append({
            "mask": m,
            "bbox": (xs.min(), ys.min(), xs.max(), ys.max()),
        })
    return objs


def filled_silhouette(mask):
    """Fill interior holes: everything not reachable from the border is inside."""
    return ndimage.binary_fill_holes(mask)


def crop(m, bbox):
    x0, y0, x1, y1 = bbox
    return m[y0:y1 + 1, x0:x1 + 1]


def iou_aligned(a, b):
    """IoU of two boolean crops aligned by their bounding-box centres."""
    H = max(a.shape[0], b.shape[0]) + 2
    W = max(a.shape[1], b.shape[1]) + 2
    A = np.zeros((H, W), bool)
    B = np.zeros((H, W), bool)
    ay, ax = (H - a.shape[0]) // 2, (W - a.shape[1]) // 2
    by, bx = (H - b.shape[0]) // 2, (W - b.shape[1]) // 2
    A[ay:ay + a.shape[0], ax:ax + a.shape[1]] = a
    B[by:by + b.shape[0], bx:bx + b.shape[1]] = b
    return (A & B).sum() / max(1, (A | B).sum())


def smoothstep(t):
    return t * t * (3 - 2 * t)


def main():
    img = np.array(Image.open(SRC).convert("RGB"))
    H, W, _ = img.shape
    nonbg = np.any(img != BG, axis=2)

    left = nonbg.copy()
    left[:, DIVIDER_X - 1:] = False
    right = nonbg.copy()
    right[:, :DIVIDER_X + 2] = False

    faces = components(left)
    outlines = components(right)
    assert len(faces) == len(outlines), (len(faces), len(outlines))

    # Silhouettes for shape matching.
    for f in faces:
        f["sil"] = crop(filled_silhouette(f["mask"]), f["bbox"])
    for o in outlines:
        o["sil"] = crop(filled_silhouette(o["mask"]), o["bbox"])

    n = len(faces)
    score = np.zeros((n, n))
    for i, f in enumerate(faces):
        for j, o in enumerate(outlines):
            score[i, j] = iou_aligned(f["sil"], o["sil"])
    best = max(itertools.permutations(range(n)), key=lambda p: sum(score[i, p[i]] for i in range(n)))

    def center(bbox):
        x0, y0, x1, y1 = bbox
        return np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])

    moves = []
    for i, f in enumerate(faces):
        o = outlines[best[i]]
        d = center(o["bbox"]) - center(f["bbox"])
        ys, xs = np.nonzero(f["mask"])
        moves.append({
            "ys": ys, "xs": xs,
            "colors": img[ys, xs],
            "delta": d,
        })
        print(f"face bbox {f['bbox']} -> outline bbox {o['bbox']}  iou={score[i, best[i]]:.3f}  delta={d}")

    # Static background: original frame with the faces erased.
    base = img.copy()
    for f in faces:
        base[f["mask"]] = BG

    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for k in range(N_FRAMES):
        s = smoothstep(k / (N_FRAMES - 1))
        fr = base.copy()
        for m in moves:
            dx, dy = np.round(m["delta"] * s).astype(int)
            ys = m["ys"] + dy
            xs = m["xs"] + dx
            ok = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
            fr[ys[ok], xs[ok]] = m["colors"][ok]
        frames.append(fr)

    # Frame 0 must equal the source exactly.
    assert np.array_equal(frames[0], img), "first frame mismatch"

    raw = np.stack(frames).tobytes()
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT,
    ]
    subprocess.run(cmd, input=raw, check=True)
    Image.fromarray(frames[-1]).save(os.path.join(OUT_DIR, "last_frame.png"))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
