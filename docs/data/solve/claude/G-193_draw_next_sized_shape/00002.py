#!/usr/bin/env python3
"""Draw the next shape in a repeating size cycle inside the empty dashed box.

The first frame shows five diamonds (small, large, medium, small, large)
followed by an empty box.  The size cycle is small -> large -> medium, so the
next shape is a medium diamond.  The video keeps every pixel outside the box
untouched and draws the medium diamond step by step: a short pause, tracing
the outline, filling from top to bottom, then holding the finished result.
"""
import os
import subprocess

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

FPS = 16
N_FRAMES = 60
SHAPE_RGB = np.array([75, 85, 99], dtype=np.uint8)
BOX_RGB = np.array([0, 0, 0], dtype=np.uint8)


def analyse(img):
    """Locate shapes, classify sizes, infer the cycle and the target box."""
    shape_mask = np.all(img == SHAPE_RGB, axis=2)
    labels, n = ndimage.label(shape_mask)
    shapes = []
    for i in range(1, n + 1):
        ys, xs = np.where(labels == i)
        shapes.append(dict(
            mask=labels == i, x0=xs.min(), x1=xs.max(), y0=ys.min(), y1=ys.max(),
            cx=(xs.min() + xs.max()) / 2.0, cy=(ys.min() + ys.max()) / 2.0,
            area=len(xs)))
    shapes.sort(key=lambda s: s["cx"])

    # Cluster sizes by area (flat colours -> exact areas per size class).
    areas = sorted(set(s["area"] for s in shapes))
    classes = []
    for a in areas:
        if classes and a - classes[-1][-1] <= max(4, 0.05 * a):
            classes[-1].append(a)
        else:
            classes.append([a])
    def size_id(s):
        for k, c in enumerate(classes):
            if s["area"] in c:
                return k
    seq = [size_id(s) for s in shapes]

    # Find shortest period p such that seq[i] == seq[i-p] for all i >= p.
    period = None
    for p in range(1, len(seq)):
        if all(seq[i] == seq[i - p] for i in range(p, len(seq))):
            period = p
            break
    if period is None:
        period = len(seq)
    next_size = seq[len(seq) - period]
    template = next(s for s in shapes if size_id(s) == next_size)

    box_mask = np.all(img == BOX_RGB, axis=2)
    ys, xs = np.where(box_mask)
    box = dict(x0=xs.min(), x1=xs.max(), y0=ys.min(), y1=ys.max())
    box["cx"] = (box["x0"] + box["x1"]) / 2.0
    box["cy"] = (box["y0"] + box["y1"]) / 2.0
    return shapes, seq, period, next_size, template, box


def build_target_mask(template, box, h, w):
    """Copy the template shape's exact pixels so it is centred in the box."""
    dx = int(round(box["cx"] - template["cx"]))
    dy = int(round(box["cy"] - template["cy"]))
    ys, xs = np.where(template["mask"])
    target = np.zeros((h, w), dtype=bool)
    target[ys + dy, xs + dx] = True
    return target, dx, dy


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0.0, 1.0))


def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    h, w, _ = base.shape
    shapes, seq, period, next_size, template, box = analyse(base)
    names = ["small", "medium", "large"]
    print("size sequence:", [names[i] if len(names) > i else i for i in seq])
    print("period:", period, "-> next size:", names[next_size])

    target, dx, dy = build_target_mask(template, box, h, w)
    ys, xs = np.where(target)
    cy, cx = ys.mean(), xs.mean()

    # Outline = boundary pixels of the target, ordered by angle around centre.
    interior = ndimage.binary_erosion(target, iterations=2)
    outline = target & ~interior
    oy, ox = np.where(outline)
    ang = (np.arctan2(oy - cy, ox - cx) + np.pi / 2) % (2 * np.pi)  # start at top
    out_frac = ang / (2 * np.pi)
    y_frac = (ys - ys.min()) / max(1, ys.max() - ys.min())

    # Timeline (frames): hold, trace outline, fill top->bottom, hold result.
    t_hold0, t_trace, t_fill = 8, 20, 24
    frames = []
    for f in range(N_FRAMES):
        img = base.copy()
        draw = np.zeros((h, w), dtype=bool)
        if f >= t_hold0:
            tr = ease((f - t_hold0 + 1) / t_trace)
            sel = out_frac <= tr + 1e-9
            draw[oy[sel], ox[sel]] = True
        if f >= t_hold0 + t_trace:
            fl = ease((f - t_hold0 - t_trace + 1) / t_fill)
            sel = y_frac <= fl + 1e-9
            draw[ys[sel], xs[sel]] = True
        if f >= t_hold0 + t_trace + t_fill:
            draw = target.copy()
        img[draw] = SHAPE_RGB
        frames.append(img)

    os.makedirs(OUT_DIR, exist_ok=True)
    raw = np.stack(frames).tobytes()
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-preset", "slow", "-crf", "8",
        "-pix_fmt", "yuv420p", "-r", str(FPS), "-movflags", "+faststart",
        OUT,
    ]
    subprocess.run(cmd, input=raw, check=True)
    print("wrote", OUT, "frames:", len(frames))


if __name__ == "__main__":
    main()
