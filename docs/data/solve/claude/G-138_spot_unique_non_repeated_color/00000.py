#!/usr/bin/env python3
"""Find the shape whose color is unique and progressively outline it in black."""
import os
import subprocess
import numpy as np
import cv2
from PIL import Image

SRC = "/app/first_frame.png"
OUT_DIR = "/app/output"
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS = 16
N_FRAMES = 21
THICKNESS = 6


def find_unique_shape(img):
    """Return the contour of the single shape whose color appears exactly once."""
    h, w, _ = img.shape
    bg = tuple(img[0, 0])  # background color sampled at the corner
    cols, counts = np.unique(img.reshape(-1, 3), axis=0, return_counts=True)
    candidates = []
    for col, cnt in zip(cols, counts):
        if tuple(col) == bg or cnt < 200:
            continue  # skip background and anti-aliasing noise
        mask = np.all(img == col, axis=2).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        comps = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 200]
        candidates.append((tuple(int(c) for c in col), len(comps), mask))
    unique = [c for c in candidates if c[1] == 1]
    assert len(unique) == 1, f"expected exactly one unique color, got {unique}"
    col, _, mask = unique[0]
    # include anti-aliased edge pixels by tolerating a small color distance
    dist = np.linalg.norm(img.astype(np.int32) - np.array(col), axis=2)
    mask = (dist < 60).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)
    return col, cnt.reshape(-1, 2)


def partial_polyline(pts, frac):
    """Points along the closed contour covering `frac` of its perimeter."""
    closed = np.vstack([pts, pts[:1]]).astype(np.float64)
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1]
    target = frac * total
    idx = np.searchsorted(cum, target)
    out = closed[:idx].tolist()
    if 0 < idx < len(closed):
        a, b = closed[idx - 1], closed[idx]
        t = (target - cum[idx - 1]) / max(seg[idx - 1], 1e-9)
        out.append((a + t * (b - a)).tolist())
    return np.array(out, dtype=np.int32)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = np.array(Image.open(SRC).convert("RGB"))
    col, cnt = find_unique_shape(base)
    print("unique color:", col, "contour points:", len(cnt))

    # start drawing from the top-most contour point for a natural look
    start = int(np.argmin(cnt[:, 1] * 10000 + cnt[:, 0]))
    cnt = np.roll(cnt, -start, axis=0)

    frames = []
    for i in range(N_FRAMES):
        frame = base.copy()
        if i == 0:
            frames.append(frame)
            continue
        frac = i / (N_FRAMES - 1)
        pts = partial_polyline(cnt, frac)
        if frac >= 1.0:
            cv2.polylines(frame, [cnt.astype(np.int32)], True, (0, 0, 0),
                          THICKNESS, lineType=cv2.LINE_AA)
        elif len(pts) >= 2:
            cv2.polylines(frame, [pts], False, (0, 0, 0), THICKNESS,
                          lineType=cv2.LINE_AA)
        frames.append(frame)

    tmp_dir = os.path.join(OUT_DIR, "_frames")
    os.makedirs(tmp_dir, exist_ok=True)
    for i, f in enumerate(frames):
        Image.fromarray(f).save(os.path.join(tmp_dir, f"{i:04d}.png"))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp_dir, "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
        "-r", str(FPS), OUT,
    ], check=True)
    for fn in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, fn))
    os.rmdir(tmp_dir)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
