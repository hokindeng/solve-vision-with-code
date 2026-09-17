#!/usr/bin/env python3
"""Move each colored object to the same-colored star marker along a straight path."""
import os, subprocess, tempfile
import numpy as np
import cv2
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 48


def components(img):
    bg = img[0, 0]
    mask = (np.abs(img.astype(int) - bg.astype(int)).sum(2) > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        m = lab == i
        cols = img[m]
        u, c = np.unique(cols.reshape(-1, 3), axis=0, return_counts=True)
        comps.append(dict(
            bbox=(x, y, w, h), area=int(area), mask=m,
            center=(x + (w - 1) / 2.0, y + (h - 1) / 2.0),
            color=tuple(int(v) for v in u[c.argmax()]),
            fill=area / float(w * h),
        ))
    return bg, comps


def is_star(c):
    # Star markers are small, sparse (low bbox fill) shapes.
    return c["fill"] < 0.35


def main():
    img = np.array(Image.open(FIRST).convert("RGB"))
    bg, comps = components(img)
    stars = [c for c in comps if is_star(c)]
    objs = [c for c in comps if not is_star(c)]

    def cdist(a, b):
        return sum((p - q) ** 2 for p, q in zip(a, b))

    # Static background: everything except the moving objects (stars stay).
    base = img.copy()
    moves = []
    for o in objs:
        star = min(stars, key=lambda s: cdist(s["color"], o["color"]))
        x, y, w, h = o["bbox"]
        patch = img[y:y + h, x:x + w].copy()
        pmask = o["mask"][y:y + h, x:x + w]
        base[o["mask"]] = bg
        start = np.array([x, y], float)
        end = start + (np.array(star["center"]) - np.array(o["center"]))
        moves.append((patch, pmask, start, end))

    H, W = img.shape[:2]
    tmp = tempfile.mkdtemp()
    for f in range(N_FRAMES):
        t = f / (N_FRAMES - 1.0)
        frame = base.copy()
        for patch, pmask, start, end in moves:
            px, py = np.round(start + (end - start) * t).astype(int)
            h, w = pmask.shape
            x0, y0 = max(px, 0), max(py, 0)
            x1, y1 = min(px + w, W), min(py + h, H)
            if x1 <= x0 or y1 <= y0:
                continue
            sub = frame[y0:y1, x0:x1]
            pm = pmask[y0 - py:y1 - py, x0 - px:x1 - px]
            sub[pm] = patch[y0 - py:y1 - py, x0 - px:x1 - px][pm]
        Image.fromarray(frame).save(os.path.join(tmp, f"{f:04d}.png"))

    os.makedirs(OUT_DIR, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp, "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT,
    ], check=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
