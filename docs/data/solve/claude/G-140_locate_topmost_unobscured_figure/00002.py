#!/usr/bin/env python3
"""Outline the topmost (unobscured) shape with a red outline, drawn progressively."""
import subprocess
import numpy as np
import cv2
from PIL import Image

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
N_FRAMES, FPS = 40, 16
RED = (255, 0, 0)
THICK = 6


def find_topmost(img):
    """Return the contour of the shape that is not occluded by any other shape.

    Each shape has a distinct flat colour. A shape hidden behind another has a
    'bite' taken out of it, so its visible area falls short of its convex hull
    and/or it splits into several components. The unobscured shape fills its
    hull as one piece.
    """
    h, w, _ = img.shape
    flat = img.reshape(-1, 3)
    cols, counts = np.unique(flat, axis=0, return_counts=True)
    bg = cols[np.argmax(counts)]
    best = None
    for c, n in zip(cols, counts):
        if n < 500 or np.array_equal(c, bg):
            continue
        mask = np.all(img == c, axis=2).astype(np.uint8)
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cs = [k for k in cs if cv2.contourArea(k) > 200]
        if not cs:
            continue
        cnt = max(cs, key=cv2.contourArea)
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        ratio = cv2.contourArea(cnt) / max(hull_area, 1)
        score = ratio - 0.5 * (len(cs) - 1)  # penalise fragmented shapes
        if best is None or score > best[0]:
            best = (score, cnt)
    return best[1]


def main():
    base = np.array(Image.open(SRC).convert("RGB"))
    cnt = find_topmost(base)
    poly = cv2.approxPolyDP(cnt, 2.0, True).reshape(-1, 2).astype(np.float64)
    # closed polyline, sampled by arc length
    pts = np.vstack([poly, poly[:1]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1]

    def partial(frac):
        L = frac * total
        out = [pts[0]]
        for i, s in enumerate(seg):
            if cum[i + 1] <= L:
                out.append(pts[i + 1])
            else:
                t = (L - cum[i]) / s if s > 0 else 0
                out.append(pts[i] + t * (pts[i + 1] - pts[i]))
                break
        return np.array(out)

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{base.shape[1]}x{base.shape[0]}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", OUT],
        stdin=subprocess.PIPE)
    for f in range(N_FRAMES):
        frame = base.copy()
        frac = f / (N_FRAMES - 1)
        if frac > 0:
            p = partial(frac)
            closed = frac >= 1.0
            cv2.polylines(frame, [np.round(p).astype(np.int32)], closed, RED, THICK, cv2.LINE_AA)
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    ff.wait()
    Image.fromarray(frame).save("/app/output/last_frame.png")


if __name__ == "__main__":
    main()
