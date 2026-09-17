#!/usr/bin/env python3
"""Move the green attention box from the left object to the right object.

The box in first_frame.png is a solid-colour, 8-px stroke with slightly rounded
corners. To reproduce its look exactly at any size, the four corner tiles and
the straight edge bands are lifted from the original frame and reassembled.
Everything else in the frame is left untouched.
"""
import subprocess, sys, os
import numpy as np
from PIL import Image

APP = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 25
BOX_RGB = np.array([70, 140, 70], np.uint8)
PAD = 34          # gap between object bbox and outer box edge (measured)
CORNER = 16       # size of the corner tile lifted from the original


def bbox(mask):
    ys, xs = np.where(mask)
    return xs.min(), ys.min(), xs.max(), ys.max()


def main():
    base = np.array(Image.open(SRC).convert("RGB"))
    H, W, _ = base.shape

    box_mask = (base == BOX_RGB).all(2)
    bx0, by0, bx1, by1 = bbox(box_mask)
    tile = box_mask[by0:by1 + 1, bx0:bx1 + 1]            # box mask, tight
    tl = tile[:CORNER, :CORNER]                          # corner tile
    top_band = tile[:CORNER, CORNER]                     # column profile of top edge
    left_band = tile[CORNER, :CORNER]                    # row profile of left edge
    # Corner symmetry is assumed: flips of the TL tile make the other corners.

    # Background with the box removed. The box sits on pure background, so
    # fill it with the dominant colour around it (white here).
    bg_colour = base[0, 0]
    clean = base.copy()
    clean[box_mask] = bg_colour

    # Objects: everything that is not background and not the box.
    obj = (clean != bg_colour).any(2)
    # Split into left / right objects by the gap between the box and the rest.
    cols = np.where(obj.any(0))[0]
    gaps = np.where(np.diff(cols) > 1)[0]
    split = (cols[gaps[0]] + cols[gaps[0] + 1]) // 2
    right = obj.copy(); right[:, :split] = False
    rx0, ry0, rx1, ry1 = bbox(right)
    target = (rx0 - PAD, ry0 - PAD, rx1 + PAD, ry1 + PAD)
    start = (bx0, by0, bx1, by1)

    def render_box(x0, y0, x1, y1):
        w, h = x1 - x0 + 1, y1 - y0 + 1
        m = np.zeros((h, w), bool)
        # straight bands
        m[:CORNER, CORNER:w - CORNER] = top_band[:, None]
        m[h - CORNER:, CORNER:w - CORNER] = top_band[::-1][:, None]
        m[CORNER:h - CORNER, :CORNER] = left_band[None, :]
        m[CORNER:h - CORNER, w - CORNER:] = left_band[::-1][None, :]
        # corners
        m[:CORNER, :CORNER] = tl
        m[:CORNER, w - CORNER:] = tl[:, ::-1]
        m[h - CORNER:, :CORNER] = tl[::-1, :]
        m[h - CORNER:, w - CORNER:] = tl[::-1, ::-1]
        full = np.zeros((H, W), bool)
        full[y0:y0 + h, x0:x0 + w] = m
        return full

    def ease(t):  # smooth ease-in-out
        return t * t * (3 - 2 * t)

    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for i in range(N_FRAMES):
        if i == 0:
            frames.append(base)
            continue
        t = ease(i / (N_FRAMES - 1))
        geom = [int(round(s + (e - s) * t)) for s, e in zip(start, target)]
        f = clean.copy()
        f[render_box(*geom)] = BOX_RGB
        frames.append(f)

    # Sanity: our renderer reproduces the original box exactly.
    assert (render_box(*start) == box_mask).all(), "box renderer mismatch"

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
           "-x264-params", "keyint=1", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        sys.exit("ffmpeg failed")
    Image.fromarray(frames[-1]).save(os.path.join(OUT_DIR, "last_frame.png"))
    print("wrote", OUT, "start box", start, "target box", target)


if __name__ == "__main__":
    main()
