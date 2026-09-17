#!/usr/bin/env python3
"""Animate a red circle being drawn around the second-largest circle."""
import os, subprocess, tempfile
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT = os.path.join(HERE, "output", "video.mp4")
FPS, N_FRAMES = 16, 40
RED = (255, 0, 0)
WIDTH = 6      # ring stroke width in px
GAP = 18       # gap between shape outline and red ring


def find_circles(im):
    """Detect filled circles: connected components of non-white, non-black colours."""
    bg = np.all(im == 255, axis=2)
    outline = np.all(im == 0, axis=2)
    fill = ~bg & ~outline
    lab, n = ndimage.label(fill)
    shapes = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if len(xs) < 200:
            continue
        cx, cy = xs.mean(), ys.mean()
        # outer radius includes the black outline
        r_fill = (xs.max() - xs.min() + 1) / 2
        r_outer = r_fill + 3
        shapes.append(dict(cx=cx, cy=cy, r=r_outer, area=len(xs)))
    return shapes


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)


def draw_arc(base, cx, cy, r, frac, ss=4):
    """Return base with a red arc of `frac` of a full circle drawn (antialiased)."""
    if frac <= 0:
        return base.copy()
    h, w = base.shape[:2]
    mask = Image.new("L", (w * ss, h * ss), 0)
    d = ImageDraw.Draw(mask)
    bbox = [(cx - r) * ss, (cy - r) * ss, (cx + r) * ss, (cy + r) * ss]
    start = -90
    end = start + 360 * min(frac, 1.0)
    if frac >= 1.0:
        d.ellipse(bbox, outline=255, width=WIDTH * ss)
    else:
        d.arc(bbox, start, end, fill=255, width=WIDTH * ss)
        # round caps
        for ang in (start, end):
            a = np.deg2rad(ang)
            px, py = (cx + (r - WIDTH / 2) * np.cos(a)) * ss, (cy + (r - WIDTH / 2) * np.sin(a)) * ss
            rr = WIDTH / 2 * ss
            d.ellipse([px - rr, py - rr, px + rr, py + rr], fill=255)
    m = np.asarray(mask.resize((w, h), Image.LANCZOS)).astype(np.float32) / 255.0
    m = m[..., None]
    out = base.astype(np.float32) * (1 - m) + np.array(RED, np.float32) * m
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    shapes = sorted(find_circles(base), key=lambda s: -s["area"])
    target = shapes[1]
    cx, cy, r = target["cx"], target["cy"], target["r"] + GAP
    print(f"target: centre=({cx:.0f},{cy:.0f}) ring radius={r:.0f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        for i in range(N_FRAMES):
            # frame 0 is untouched; ring completes on the second-to-last frame, last frame holds
            t = i / (N_FRAMES - 2) if i < N_FRAMES - 1 else 1.0
            frame = draw_arc(base, cx, cy, r, ease(min(t, 1.0)))
            Image.fromarray(frame).save(os.path.join(td, f"f{i:03d}.png"))
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
            "-i", os.path.join(td, "f%03d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
            "-r", str(FPS), OUT,
        ], check=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
