#!/usr/bin/env python3
"""Move each colored object onto the star marker of the same color.

Detects the background color, groups non-background pixels by color, splits
each color into connected components, classifies components as "object"
(high bounding-box fill ratio) or "star marker" (low fill ratio), then renders
48 frames translating each object along a straight line so its center lands
on its marker's center. Stars and background are untouched.
"""
import os
import shutil
import subprocess
import tempfile
import numpy as np
from PIL import Image
from scipy import ndimage

SRC = "/app/first_frame.png"
OUT_DIR = "/app/output"
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS = 16
N_FRAMES = 48


def load():
    return np.array(Image.open(SRC).convert("RGB"))


def background_color(img):
    cols, counts = np.unique(img.reshape(-1, 3), axis=0, return_counts=True)
    return tuple(cols[np.argmax(counts)])


def components_by_color(img, bg):
    """Return dict color -> list of (mask, bbox_center)."""
    flat = img.reshape(-1, 3)
    cols, counts = np.unique(flat, axis=0, return_counts=True)
    result = {}
    for c, n in zip(cols, counts):
        c = tuple(int(v) for v in c)
        if c == tuple(int(v) for v in bg) or n < 30:
            continue
        mask = np.all(img == np.array(c, dtype=np.uint8), axis=2)
        lab, k = ndimage.label(mask)
        comps = []
        for i in range(1, k + 1):
            m = lab == i
            if m.sum() < 30:
                continue
            ys, xs = np.nonzero(m)
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            area = m.sum()
            bbox_area = (y1 - y0 + 1) * (x1 - x0 + 1)
            fill = area / bbox_area
            center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
            comps.append(dict(mask=m, center=center, fill=fill, area=area))
        result[c] = comps
    return result


def classify(comps):
    """Split one color's components into (object, star)."""
    # Stars have a low bbox fill ratio (thin spikes); solid objects are high.
    comps = sorted(comps, key=lambda d: d["fill"])
    star, obj = comps[0], comps[-1]
    return obj, star


def render(img, bg, moves, t):
    """moves: list of (obj_mask, obj_color, dx_total, dy_total)."""
    frame = img.copy()
    # Erase objects from their original positions.
    for m, color, dx, dy in moves:
        frame[m] = bg
    # Paste objects at interpolated positions.
    for m, color, dx, dy in moves:
        ox = int(round(dx * t))
        oy = int(round(dy * t))
        ys, xs = np.nonzero(m)
        ys2 = ys + oy
        xs2 = xs + ox
        ok = (ys2 >= 0) & (ys2 < frame.shape[0]) & (xs2 >= 0) & (xs2 < frame.shape[1])
        frame[ys2[ok], xs2[ok]] = color
    return frame


def main():
    img = load()
    bg = background_color(img)
    groups = components_by_color(img, bg)
    moves = []
    for color, comps in groups.items():
        if len(comps) < 2:
            continue
        obj, star = classify(comps)
        dx = star["center"][0] - obj["center"][0]
        dy = star["center"][1] - obj["center"][1]
        moves.append((obj["mask"], np.array(color, dtype=np.uint8), dx, dy))
        print(f"color {color}: object center {obj['center']} -> star center {star['center']}")

    os.makedirs(OUT_DIR, exist_ok=True)
    frames_dir = tempfile.mkdtemp(prefix="frames_")
    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)  # linear pacing, arrives on last frame
        frame = img if i == 0 else render(img, bg, moves, t)
        Image.fromarray(frame).save(os.path.join(frames_dir, f"{i:04d}.png"))

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(frames_dir, "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT,
    ], check=True)
    shutil.rmtree(frames_dir, ignore_errors=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
