#!/usr/bin/env python3
"""Circle the single asymmetrical shape in first_frame.png with a red circle,
animated as an arc that sweeps around over 16 frames."""
import os, subprocess, tempfile
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 16


def find_asymmetric_shape(img):
    """Return (cx, cy, radius) for the shape whose silhouette is least symmetric."""
    mask = (np.abs(img.astype(int) - 255).sum(2) > 30).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    best, best_score = None, -1.0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 200:
            continue
        m = (labels[y:y + h, x:x + w] == i).astype(np.uint8)
        # Symmetry score: max overlap with its own mirror across vertical,
        # horizontal axes, and 180 deg rotation. Asymmetric shape = lowest max.
        cands = [m[:, ::-1], m[::-1, :], m[::-1, ::-1]]
        sym = max((m & c).sum() / max(1, (m | c).sum()) for c in cands)
        score = 1.0 - sym
        if score > best_score:
            best_score, best = score, (x, y, w, h)
    x, y, w, h = best
    cx, cy = x + w / 2.0, y + h / 2.0
    r = 0.5 * float(np.hypot(w, h)) + 8.0
    return cx, cy, r


def main():
    base = cv2.imread(SRC)
    assert base is not None and base.shape[:2] == (1024, 1024)
    cx, cy, r = find_asymmetric_shape(base)
    os.makedirs(OUT_DIR, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        for k in range(N_FRAMES):
            f = base.copy()
            t = k / (N_FRAMES - 1)          # 0 .. 1
            if t > 0:
                sweep = 360.0 * t
                cv2.ellipse(f, (int(round(cx)), int(round(cy))),
                            (int(round(r)), int(round(r))), 0, -90, -90 + sweep,
                            (0, 0, 255), 5, lineType=cv2.LINE_AA)
            cv2.imwrite(os.path.join(td, f"f{k:03d}.png"), f)
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
            "-i", os.path.join(td, "f%03d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
            "-r", str(FPS), OUT], check=True)
    print(f"wrote {OUT}  circle center=({cx:.1f},{cy:.1f}) r={r:.1f}")


if __name__ == "__main__":
    main()
