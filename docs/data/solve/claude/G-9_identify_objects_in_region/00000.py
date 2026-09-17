#!/usr/bin/env python3
"""Outline all circles inside the circular (black-bordered) region with a green border.

Reads /app/first_frame.png, detects the circular region and the circle-shaped
objects inside it, and animates a green outline being swept around each one
over 40 frames at 16 fps. Everything else is left pixel-identical.
"""
import os
import subprocess
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 40, 16
GREEN = (0, 200, 0)          # RGB
THICK = 4                    # outline thickness in px


def find_circular_region(img):
    """Return (cx, cy, r) of the large circle drawn with a black border."""
    black = np.all(img < 40, axis=2).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(black, 8)
    best = None
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w < 100 or h < 100:
            continue
        # fill the border to compare area with an ideal disk -> circle vs square
        ys, xs = np.nonzero(lab == i)
        pts = np.stack([xs, ys], 1).astype(np.float32)
        (cx, cy), r = cv2.minEnclosingCircle(pts)
        d = np.hypot(xs - cx, ys - cy)
        # for a circle border all points lie near radius r; for a square they don't
        if d.std() < 2.0 and (best is None or w * h > best[0]):
            best = (w * h, cx, cy, r)
    if best is None:
        raise RuntimeError("circular region not found")
    return best[1:]


def find_circles_in_region(img, region):
    """Return list of (cx, cy, r) for circle-shaped objects inside the region."""
    cx0, cy0, r0 = region
    nonwhite = np.any(img < 235, axis=2).astype(np.uint8)
    # drop the region border itself
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    inside = np.hypot(xx - cx0, yy - cy0) < r0 - 4
    nonwhite &= inside.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(nonwhite, 8)
    found = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 100:
            continue
        m = (lab == i).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cnts, key=cv2.contourArea)
        a = cv2.contourArea(c)
        p = cv2.arcLength(c, True)
        circularity = 4 * np.pi * a / (p * p) if p > 0 else 0
        (cx, cy), r = cv2.minEnclosingCircle(c)
        fill_ratio = a / (np.pi * r * r)
        if circularity > 0.85 and fill_ratio > 0.85:
            found.append((float(cx), float(cy), float(r)))
    return found


def draw_outline(frame, circ, frac):
    """Draw a green arc covering `frac` of the full outline (anti-aliased)."""
    if frac <= 0:
        return
    cx, cy, r = circ
    rr = r + THICK / 2.0 - 1.0  # ring hugs the object's outer edge
    S = 4  # supersample for smooth edges
    x0, y0 = int(cx - rr - THICK - 2), int(cy - rr - THICK - 2)
    x1, y1 = int(cx + rr + THICK + 3), int(cy + rr + THICK + 3)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, frame.shape[1]), min(y1, frame.shape[0])
    W, H = (x1 - x0) * S, (y1 - y0) * S
    layer = np.zeros((H, W), np.uint8)
    center = (int(round((cx - x0) * S)), int(round((cy - y0) * S)))
    cv2.ellipse(layer, center, (int(round(rr * S)), int(round(rr * S))),
                0, -90, -90 + 360 * min(frac, 1.0), 255, int(THICK * S), cv2.LINE_AA)
    alpha = cv2.resize(layer, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    g = np.array(GREEN, np.float32)
    roi = roi * (1 - alpha[..., None]) + g * alpha[..., None]
    frame[y0:y1, x0:x1] = np.clip(roi + 0.5, 0, 255).astype(np.uint8)


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = cv2.cvtColor(cv2.imread(SRC), cv2.COLOR_BGR2RGB)
    region = find_circular_region(base)
    circles = find_circles_in_region(base, region)
    print("region:", region, "circles:", circles)

    tmp = os.path.join(OUT_DIR, "frames")
    os.makedirs(tmp, exist_ok=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    for k in range(N_FRAMES):
        frame = base.copy()
        # frame 0 is untouched; the outline sweeps in over the remaining frames
        t = 0.0 if k == 0 else ease(k / (N_FRAMES - 1))
        for c in circles:
            draw_outline(frame, c, t)
        cv2.imwrite(os.path.join(tmp, f"{k:03d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp, "%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT,
    ], check=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
