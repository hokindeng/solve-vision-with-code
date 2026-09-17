#!/usr/bin/env python3
"""Generate video: circle the middle dot (by count) of a row of dots with a red circle."""
import os, subprocess, tempfile
import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT = os.path.join(HERE, "output", "video.mp4")
FPS, N_FRAMES = 16, 22
RED = (220, 30, 30)

def find_dots(img):
    gray = np.array(img.convert("L"))
    lab, n = ndimage.label(gray < 128)
    dots = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if len(xs) < 50:
            continue
        dots.append(((xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0,
                     (xs.max() - xs.min() + 1) / 2.0))
    dots.sort(key=lambda d: d[0])
    return dots

def draw_arc(base, cx, cy, r, width, sweep_deg, color, ss=4):
    """Anti-aliased red arc from -90deg sweeping clockwise `sweep_deg`, composited onto base (RGB uint8)."""
    if sweep_deg <= 0:
        return base
    h, w, _ = base.shape
    pad = int(r + width + 3)
    x0, x1 = max(0, int(cx) - pad), min(w, int(cx) + pad + 1)
    y0, y1 = max(0, int(cy) - pad), min(h, int(cy) + pad + 1)
    # supersampled coverage mask
    yy, xx = np.mgrid[y0:y1, x0:x1]
    yy = (yy[:, :, None, None] + (np.arange(ss)[None, None, :, None] + 0.5) / ss) - cy
    xx = (xx[:, :, None, None] + (np.arange(ss)[None, None, None, :] + 0.5) / ss) - cx
    dist = np.sqrt(xx ** 2 + yy ** 2)
    ring = np.abs(dist - r) <= width / 2.0
    ang = (np.degrees(np.arctan2(yy, xx)) + 90.0) % 360.0  # 0 at top, clockwise
    inarc = ang <= sweep_deg
    cov = (ring & inarc).mean(axis=(2, 3))
    # round end caps
    for a in (0.0, sweep_deg):
        t = np.radians(a - 90.0)
        ex, ey = cx + r * np.cos(t), cy + r * np.sin(t)
        cap = (np.sqrt((xx + cx - ex) ** 2 + (yy + cy - ey) ** 2) <= width / 2.0).mean(axis=(2, 3))
        cov = np.maximum(cov, cap)
    out = base.copy()
    region = out[y0:y1, x0:x1].astype(np.float32)
    col = np.array(color, dtype=np.float32)
    region = region * (1 - cov[..., None]) + col * cov[..., None]
    out[y0:y1, x0:x1] = np.clip(region + 0.5, 0, 255).astype(np.uint8)
    return out

def main():
    img = Image.open(FIRST).convert("RGB")
    base = np.array(img)
    dots = find_dots(img)
    assert len(dots) % 2 == 1, "need odd number of dots"
    cx, cy, dr = dots[len(dots) // 2]
    gap = min(dots[i + 1][0] - dots[i][0] for i in range(len(dots) - 1)) if len(dots) > 1 else 4 * dr
    width = 4.0
    r = min(dr + 7.0, (gap - dr) - width / 2 - 2)  # stay clear of neighbours
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        for i in range(N_FRAMES):
            if i == 0:
                frame = base
            else:
                t = i / (N_FRAMES - 1)
                t = t * t * (3 - 2 * t)  # ease in/out
                frame = draw_arc(base, cx, cy, r, width, 360.0 * t, RED)
            Image.fromarray(frame).save(os.path.join(td, f"f{i:03d}.png"))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                        "-i", os.path.join(td, "f%03d.png"), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT], check=True)
    print("wrote", OUT)

if __name__ == "__main__":
    main()
