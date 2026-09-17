#!/usr/bin/env python3
"""Find the single pentagon in first_frame.png and mark it with a red circle
that expands from its centre outward until it encircles the shape."""
import os
import subprocess
import numpy as np
import cv2
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 30, 16
RED = np.array([220, 30, 30], dtype=np.float32)
THICKNESS = 6.0        # ring thickness in pixels
MARGIN = 18.0          # gap between shape and final ring


def find_pentagon(img):
    """Return (cx, cy, radius_needed) of the polygon with exactly 5 vertices."""
    bg = img[0, 0].astype(int)
    mask = (np.abs(img.astype(int) - bg).sum(axis=2) > 40).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        if cv2.contourArea(c) < 200:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 5:
            if best is not None:
                raise RuntimeError("more than one pentagon found")
            best = c
    if best is None:
        raise RuntimeError("no pentagon found")
    M = cv2.moments(best)
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    pts = best.reshape(-1, 2).astype(np.float64)
    r = np.sqrt(((pts - (cx, cy)) ** 2).sum(axis=1)).max()
    return cx, cy, r


def draw_ring(base, cx, cy, radius, alpha_scale=1.0):
    """Anti-aliased red ring of given radius composited over base (float32)."""
    if radius <= 0:
        return base
    h, w = base.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    half = THICKNESS / 2.0
    # coverage: 1 inside the ring band, smooth 1px falloff at edges
    a = np.clip(half - np.abs(d - radius) + 0.5, 0.0, 1.0) * alpha_scale
    a = a[..., None]
    return base * (1 - a) + RED * a


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    img = np.array(Image.open(SRC).convert("RGB"))
    cx, cy, r_shape = find_pentagon(img)
    r_final = r_shape + MARGIN
    base = img.astype(np.float32)

    frames = []
    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)
        # ease-out so the ring settles gently at the end
        e = 1 - (1 - t) ** 2
        radius = e * r_final
        if i == 0:
            frame = base  # first frame must be untouched
        else:
            frame = draw_ring(base, cx, cy, radius)
        frames.append(np.clip(frame + 0.5, 0, 255).astype(np.uint8))

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
    for name in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, name))
    os.rmdir(tmp_dir)
    print(f"pentagon at ({cx:.1f}, {cy:.1f}), r={r_shape:.1f}; wrote {OUT}")


if __name__ == "__main__":
    main()
