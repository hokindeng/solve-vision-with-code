"""Outline all triangles inside the circular region with a green border.

Approach: detect the black circle boundary to get the region; detect colored
shapes via contour analysis; keep triangles whose centroid lies inside the
circle; animate a green outline tracing each triangle's perimeter.
"""
import os, subprocess
import numpy as np
import cv2
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(BASE, "first_frame.png")
OUT_DIR = os.path.join(BASE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 40, 16
GREEN = (0, 200, 0)          # RGB
THICK = 5


def find_circle(img):
    """Locate the large black-bordered circle; return (cx, cy, r)."""
    dark = (img.max(axis=2) < 40).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        (x, y), r = cv2.minEnclosingCircle(c)
        area = cv2.contourArea(c)
        if r < 50:
            continue
        # circle: enclosed area close to pi r^2 (contour is the ring's outer edge)
        fill = area / (np.pi * r * r)
        if fill > 0.9 and (best is None or r > best[2]):
            best = (x, y, r)
    return best


def find_triangles(img):
    """Return list of 3-vertex polygons (Nx2 int) for colored triangles."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = ((hsv[..., 1] > 60) & (hsv[..., 2] > 60)).astype(np.uint8) * 255
    # include the dark (80,80,80) shape outlines so contours cover the full shape
    outline = (np.abs(img.astype(int) - 80).max(axis=2) < 15).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask | outline, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tris = []
    for c in cnts:
        if cv2.contourArea(c) < 300:
            continue
        approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
        if len(approx) == 3:
            tris.append(approx.reshape(-1, 2))
    return tris


def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    cx, cy, r = find_circle(base)
    tris = [t for t in find_triangles(base)
            if np.hypot(t[:, 0].mean() - cx, t[:, 1].mean() - cy) < r]
    assert tris, "no triangle found inside the circle"

    # perimeter points for progressive tracing
    def perimeter(t):
        pts = [tuple(map(float, p)) for p in t]
        segs = [(pts[i], pts[(i + 1) % 3]) for i in range(3)]
        lens = [np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in segs]
        return segs, lens, sum(lens)

    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for i in range(N_FRAMES):
        t = 1.0 if i == N_FRAMES - 1 else i / (N_FRAMES - 1)
        t = t * t * (3 - 2 * t)  # ease in/out
        f = base.copy()
        if i > 0:
            for tri in tris:
                segs, lens, total = perimeter(tri)
                remaining = t * total
                pts = [tuple(np.round(segs[0][0]).astype(int))]
                for (a, b), L in zip(segs, lens):
                    if remaining <= 0:
                        break
                    frac = min(1.0, remaining / L)
                    end = (a[0] + (b[0] - a[0]) * frac, a[1] + (b[1] - a[1]) * frac)
                    pts.append(tuple(np.round(end).astype(int)))
                    remaining -= L
                closed = t >= 1.0
                cv2.polylines(f, [np.array(pts, np.int32)], closed, GREEN, THICK,
                              lineType=cv2.LINE_AA)
        frames.append(f)

    # Write via ffmpeg for exact codec/pixel format control
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "1024x1024", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()
    Image.fromarray(frames[-1]).save(os.path.join(OUT_DIR, "last_frame.png"))
    print(f"circle=({cx:.0f},{cy:.0f},r={r:.0f}) triangles={len(tris)} -> {OUT}")


if __name__ == "__main__":
    main()
