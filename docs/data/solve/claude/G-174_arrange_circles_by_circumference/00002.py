#!/usr/bin/env python3
"""Rearrange the 7 circles of first_frame.png into a centered horizontal row,
sorted left-to-right by circumference (largest first), and render the motion
as an 80-frame, 16 fps, 1024x1024 H.264 video."""
import os
import subprocess
import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 80, 16
GAP = 30  # horizontal spacing between neighbouring circles in the final row


def detect_circles(img):
    """Return list of sprites: dict(patch, mask, x0, y0, r) via connected components."""
    mask = (img != 255).any(axis=2)
    lab, n = ndimage.label(mask)
    sprites = []
    for i in range(1, n + 1):
        m = lab == i
        ys, xs = np.nonzero(m)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        patch = img[y0:y1 + 1, x0:x1 + 1].copy()
        pm = m[y0:y1 + 1, x0:x1 + 1]
        # fill any interior holes so the sprite is a solid disc
        pm = ndimage.binary_fill_holes(pm)
        w, h = x1 - x0 + 1, y1 - y0 + 1
        sprites.append(dict(patch=patch, mask=pm, x0=int(x0), y0=int(y0),
                            w=int(w), h=int(h), r=(w + h) / 4.0))
    return sprites


def ease(t):
    # smooth ease-in-out (cosine)
    return 0.5 - 0.5 * np.cos(np.pi * t)


def main():
    img = np.array(Image.open(SRC).convert("RGB"))
    H, W = img.shape[:2]
    sprites = detect_circles(img)

    # background: original image with the circles removed (white behind them)
    bg = img.copy()
    for s in sprites:
        region = bg[s["y0"]:s["y0"] + s["h"], s["x0"]:s["x0"] + s["w"]]
        region[s["mask"]] = 255

    # sort by circumference (2*pi*r) descending == radius descending
    order = sorted(range(len(sprites)), key=lambda i: -sprites[i]["r"])
    total_w = sum(sprites[i]["w"] for i in order) + GAP * (len(order) - 1)
    x = (W - total_w) / 2.0
    cy = H / 2.0
    for i in order:
        s = sprites[i]
        s["tx0"] = x
        s["ty0"] = cy - s["h"] / 2.0
        x += s["w"] + GAP

    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for f in range(N_FRAMES):
        t = ease(f / (N_FRAMES - 1))
        frame = bg.copy()
        # draw larger circles first so smaller ones stay visible while passing
        for i in order:
            s = sprites[i]
            px = int(round(s["x0"] + (s["tx0"] - s["x0"]) * t))
            py = int(round(s["y0"] + (s["ty0"] - s["y0"]) * t))
            region = frame[py:py + s["h"], px:px + s["w"]]
            region[s["mask"]] = s["patch"][s["mask"]]
        frames.append(frame)
    frames[0] = img.copy()  # first frame identical to the source

    # write via ffmpeg (raw rgb24 pipe -> libx264 yuv420p)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "medium", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(np.ascontiguousarray(fr).tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print("wrote", OUT, "frames", len(frames))


if __name__ == "__main__":
    main()
