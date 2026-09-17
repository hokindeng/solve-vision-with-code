#!/usr/bin/env python3
"""Locate the intersection of two line segments in first_frame.png and animate
drawing a single red circle around it. Output: /app/output/video.mp4."""
import subprocess
import numpy as np
import cv2
from PIL import Image

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
N_FRAMES = 30
FPS = 16
RED = (220, 30, 30)
RADIUS = 42
THICK = 5


def fit_line(mask):
    """PCA line fit through mask pixels -> (point, unit direction)."""
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    return c, vt[0]


def intersect(p1, d1, p2, d2):
    # p1 + t*d1 = p2 + s*d2
    A = np.array([d1, -d2]).T
    t, _ = np.linalg.solve(A, p2 - p1)
    return p1 + t * d1


def main():
    base = np.array(Image.open(SRC).convert("RGB"))
    h, w, _ = base.shape

    # Segment the two coloured lines: non-white pixels clustered by colour.
    nonwhite = np.any(base < 235, axis=2)
    cols = base[nonwhite].reshape(-1, 3).astype(np.float32)
    _, labels, centers = cv2.kmeans(cols, 2, None,
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1),
                                    5, cv2.KMEANS_PP_CENTERS)
    lab_img = np.full((h, w), -1, np.int32)
    lab_img[nonwhite] = labels.ravel()
    # Antialiased edge pixels blend towards white; keep only confident ones.
    masks = []
    for k in range(2):
        m = lab_img == k
        dist = np.linalg.norm(base.astype(np.float32) - centers[k], axis=2)
        masks.append(m & (dist < 60))

    p1, d1 = fit_line(masks[0])
    p2, d2 = fit_line(masks[1])
    ix, iy = intersect(p1, d1, p2, d2)
    print(f"intersection at ({ix:.2f}, {iy:.2f})")

    center = (float(ix), float(iy))
    frames = []
    for i in range(N_FRAMES):
        f = base.copy()
        if i > 0:
            # Sweep the circle progressively; complete on the last frame.
            t = min(1.0, i / (N_FRAMES - 2))
            sweep = 360.0 * t
            if sweep >= 359.9:
                cv2.circle(f, (int(round(ix)), int(round(iy))), RADIUS, RED, THICK, cv2.LINE_AA)
            else:
                cv2.ellipse(f, (int(round(ix)), int(round(iy))), (RADIUS, RADIUS),
                            0, -90, -90 + sweep, RED, THICK, cv2.LINE_AA)
        frames.append(f)

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
