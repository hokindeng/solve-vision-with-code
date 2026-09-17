#!/usr/bin/env python3
"""Animate the refracted ray for the air-glass Snell's-law scene.

Frame 0 is /app/first_frame.png. Over the video the angle annotation
(arc + "θ = 42°" label) fades away and the red refracted ray grows from the
incidence point to the image boundary. Every other pixel is untouched.
"""
import math
import os
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

FPS = 16
N_FRAMES = 70
W = H = 1024

THETA1_DEG = 41.7
N_AIR, N_GLASS = 1.00, 1.387
RAY_WIDTH = 4                      # matches the blue incident ray
RED = (255, 0, 0)
BLUE = (0, 0, 255)


def locate_scene(a):
    """Find the interface row, incidence point and incident-ray footprint."""
    dark = (a.max(axis=2) < 100)
    interface_rows = [y for y in range(H) if dark[y].sum() > W // 2]
    y_if_top, y_if_bot = min(interface_rows), max(interface_rows)
    y_if = (y_if_top + y_if_bot) / 2.0

    blue = (a[:, :, 2] > 150) & (a[:, :, 0] < 100) & (a[:, :, 1] < 100)
    # normal line (gray 150) gives the x of the incidence point
    gray = (a[:, :, 0] == 150) & (a[:, :, 1] == 150) & (a[:, :, 2] == 150)
    xs = np.nonzero(gray.any(axis=0))[0]
    x0 = int(round(xs.mean())) if len(xs) else int(np.nonzero(blue[y_if_top - 1])[0].mean())
    return y_if_top, y_if_bot, y_if, x0, blue, dark


def clean_annotation(a):
    """Return the frame with the angle annotation removed, plus a change mask."""
    y_if_top, y_if_bot, y_if, x0, blue, dark = locate_scene(a)
    # annotation = anything above the interface that is not white, not the
    # blue incident ray and not the gray normal line (includes anti-aliased
    # edge pixels of the arc and the text label)
    white = (a == 255).all(axis=2)
    normal = (a[:, :, 0] == 150) & (a[:, :, 1] == 150) & (a[:, :, 2] == 150)
    ann = ~white & ~blue & ~normal
    ann[y_if_top:] = False              # keep interface line and everything below

    # Rows where the incident ray is not touched by annotation pixels
    ann_near_ray = np.zeros(H, bool)
    for y in range(y_if_top):
        bx = np.nonzero(blue[y])[0]
        ax = np.nonzero(ann[y])[0]
        if len(bx) and len(ax):
            if (np.abs(ax[:, None] - bx[None, :]).min() <= 1):
                ann_near_ray[y] = True

    def run(y):
        bx = np.nonzero(blue[y])[0]
        return (bx.min(), bx.max()) if len(bx) else None

    out = a.copy()
    ys, xs = np.nonzero(ann)
    for x, y in zip(xs, ys):
        new = (255, 255, 255)
        if ann_near_ray[y]:
            # interpolate the blue run from the nearest clean rows above/below
            ya = y - 1
            while ya >= 0 and (ann_near_ray[ya] or run(ya) is None):
                ya -= 1
            yb = y + 1
            while yb < y_if_top and (ann_near_ray[yb] or run(yb) is None):
                yb += 1
            ra, rb = run(ya), run(yb)
            if ra is not None and rb is not None:
                t = (y - ya) / (yb - ya)
                lo = ra[0] + t * (rb[0] - ra[0])
                hi = ra[1] + t * (rb[1] - ra[1])
                if lo - 0.5 <= x <= hi + 0.5:
                    new = BLUE
        out[y, x] = new
    return out, ann, (x0, y_if)


def ray_end(x0, y0):
    th1 = math.radians(THETA1_DEG)
    th2 = math.asin(N_AIR * math.sin(th1) / N_GLASS)
    dx, dy = math.sin(th2), math.cos(th2)       # downward-right, same side as incident
    # extend to the image boundary
    t_bottom = (H - 1 - y0) / dy
    t_right = (W - 1 - x0) / dx if dx > 0 else float("inf")
    t = min(t_bottom, t_right)
    return (x0 + t * dx, y0 + t * dy), t


def draw_ray(base, x0, y0, end, frac):
    if frac <= 0:
        return base
    img = Image.fromarray(base)
    ex = x0 + (end[0] - x0) * frac
    ey = y0 + (end[1] - y0) * frac
    ImageDraw.Draw(img).line([(x0, y0), (ex, ey)], fill=RED, width=RAY_WIDTH)
    return np.array(img)


def main():
    a = np.array(Image.open(SRC).convert("RGB"))
    cleaned, ann_mask, (x0, y_if) = clean_annotation(a)
    y0 = y_if
    end, _ = ray_end(x0, y0)

    fade_start, fade_end = 1, 20      # annotation fades out
    grow_start, grow_end = 12, N_FRAMES - 1   # ray grows to the boundary

    frames = []
    for i in range(N_FRAMES):
        # annotation fade: blend original -> cleaned on the annotation pixels only
        if i < fade_start:
            frame = a.copy()
        elif i >= fade_end:
            frame = cleaned.copy()
        else:
            s = (i - fade_start + 1) / (fade_end - fade_start + 1)
            frame = a.copy()
            frame[ann_mask] = np.round(
                a[ann_mask] * (1 - s) + cleaned[ann_mask] * s).astype(np.uint8)
        # ray growth
        if i >= grow_start:
            frac = (i - grow_start) / (grow_end - grow_start)
            frac = min(1.0, max(0.0, frac))
            frame = draw_ray(frame, x0, y0, end, frac)
        frames.append(frame)

    os.makedirs(OUT_DIR, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        for i, f in enumerate(frames):
            Image.fromarray(f).save(os.path.join(td, f"f{i:04d}.png"))
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
            "-i", os.path.join(td, "f%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "10",
            "-preset", "slow", "-r", str(FPS), OUT,
        ], check=True)
    Image.fromarray(frames[-1]).save(os.path.join(OUT_DIR, "last_frame.png"))
    print(f"wrote {OUT}: {N_FRAMES} frames @ {FPS} fps; ray end {end}")


if __name__ == "__main__":
    main()
