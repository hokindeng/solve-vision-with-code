#!/usr/bin/env python3
"""Regenerate /app/output/video.mp4: progressively trace a red outline around the
topmost (unobscured) shape in first_frame.png, the blue triangle."""
import os, subprocess, shutil, tempfile
import numpy as np
import cv2
from PIL import Image

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

N_FRAMES = 40
FPS = 16
RED = (255, 0, 0)
THICK = 6

def _polygon(mask, eps=2.5):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    return cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float64)

def _edges(poly):
    return [(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]

def _is_subsegment(e, f, tol=3.0, margin=6.0):
    """True if edge e lies on the line of edge f and f extends beyond e."""
    a, b = e
    p, q = f
    d = q - p
    L = np.linalg.norm(d)
    if L < 1e-6:
        return False
    u = d / L
    n = np.array([-u[1], u[0]])
    if abs(np.dot(a - p, n)) > tol or abs(np.dot(b - p, n)) > tol:
        return False
    ta, tb = np.dot(a - p, u), np.dot(b - p, u)
    lo, hi = min(ta, tb), max(ta, tb)
    if lo < -tol or hi > L + tol:          # e must sit inside f
        return False
    return (lo > margin) or (hi < L - margin)   # f strictly longer somewhere

def find_topmost(im):
    """Return the polygon of the shape no other shape occludes.

    Where shape A is occluded by shape B, the visible outline of A contains a
    piece of B's straight edge that is shorter than B's full edge (a
    T-junction). A shape with no such edge is on top."""
    bg = tuple(im[0, 0])
    colors = [tuple(c) for c in np.unique(im.reshape(-1, 3), axis=0) if tuple(c) != bg]
    polys = {c: _polygon(np.all(im == c, axis=2)) for c in colors}
    under = {c: 0 for c in colors}
    for a in colors:
        for b in colors:
            if a == b:
                continue
            for e in _edges(polys[a]):
                if any(_is_subsegment(e, f) for f in _edges(polys[b])):
                    under[a] += 1
    top = min(colors, key=lambda c: under[c])
    # Unobscured shape: its visible region is its full (convex) outline.
    hull = cv2.convexHull(polys[top].astype(np.float32)).reshape(-1, 2)
    return cv2.approxPolyDP(hull, 2.5, True).reshape(-1, 2).astype(np.float64), under

def path_points(poly, frac):
    """Points along the closed polygon perimeter up to fraction frac in [0,1]."""
    pts = np.vstack([poly, poly[:1]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = seg.sum()
    target = frac * total
    out = [pts[0]]
    acc = 0.0
    for i, L in enumerate(seg):
        if acc + L >= target:
            t = (target - acc) / L
            out.append(pts[i] + t * (pts[i + 1] - pts[i]))
            break
        out.append(pts[i + 1])
        acc += L
    return np.array(out)

def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    poly, _ = find_topmost(base)
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = tempfile.mkdtemp()
    for i in range(N_FRAMES):
        frame = base.copy()
        frac = i / (N_FRAMES - 1)
        if frac > 0:
            pts = path_points(poly, frac)
            closed = frac >= 1.0
            cv2.polylines(frame, [np.round(pts).astype(np.int32)], closed, RED, THICK, cv2.LINE_AA)
        Image.fromarray(frame).save(os.path.join(tmp, f"f{i:03d}.png"))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp, "f%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-movflags", "+faststart", OUT], check=True)
    shutil.rmtree(tmp)
    print("wrote", OUT)

if __name__ == "__main__":
    main()
