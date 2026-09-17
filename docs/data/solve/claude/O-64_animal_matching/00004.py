#!/usr/bin/env python3
"""Move each colored animal face (left) onto its matching outline (right)."""
import subprocess
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
W = H = 1024
FPS = 16
N_FRAMES = 64
DIVIDER_X = 512


def components(mask, x0, x1):
    """Connected components (after dilation to merge parts) restricted to x range."""
    m = np.zeros_like(mask)
    m[:, x0:x1] = mask[:, x0:x1]
    lab, n = ndimage.label(ndimage.binary_dilation(m, iterations=6))
    comps = []
    for i in range(1, n + 1):
        cm = ndimage.binary_fill_holes((lab == i) & m)  # include near-bg interior pixels (e.g. white teeth)
        ys, xs = np.where(cm)
        comps.append(dict(mask=cm, x0=xs.min(), x1=xs.max(), y0=ys.min(), y1=ys.max()))
    return comps


def center(c):
    return np.array([(c["x0"] + c["x1"]) / 2.0, (c["y0"] + c["y1"]) / 2.0])


def size(c):
    return np.array([c["x1"] - c["x0"], c["y1"] - c["y0"]], float)


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)  # smooth in/out


def main():
    base = np.array(Image.open(SRC).convert("RGB"))
    bg = base[5, 5].astype(int)
    diff = np.abs(base.astype(int) - bg).sum(2)
    mask = diff > 20

    faces = components(mask, 0, DIVIDER_X - 6)
    outlines = components(mask, DIVIDER_X + 6, W)

    # match faces to outlines by bounding-box size (shapes are identical)
    cost = np.array([[np.abs(size(f) - size(o)).sum() for o in outlines] for f in faces])
    ri, ci = linear_sum_assignment(cost)

    # background plate: faces erased
    plate = base.copy()
    sprites = []
    for fi, oi in zip(ri, ci):
        f, o = faces[fi], outlines[oi]
        m = f["mask"]
        ys, xs = np.where(m)
        plate[m] = bg
        sub = (slice(f["y0"], f["y1"] + 1), slice(f["x0"], f["x1"] + 1))
        sprites.append(dict(
            rgb=base[sub].copy(),
            mask=m[sub].copy(),
            start=np.array([f["x0"], f["y0"]], float),
            delta=center(o) - center(f),
        ))

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-r", str(FPS), OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for k in range(N_FRAMES):
        t = ease(k / (N_FRAMES - 1))
        frame = base.copy() if k == 0 else plate.copy()
        if k > 0:
            for s in sprites:
                pos = np.rint(s["start"] + t * s["delta"]).astype(int)
                x, y = pos
                h, w = s["mask"].shape
                region = frame[y:y + h, x:x + w]
                region[s["mask"]] = s["rgb"][s["mask"]]
        proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    proc.stdin.close()
    proc.wait()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
