#!/usr/bin/env python3
"""Move two ring-shaped circles so they become concentric at the image center.

The circle sprites are lifted pixel-exactly from first_frame.png (the orange
circle is clipped at the top edge, so its missing part is rebuilt by mirror
symmetry, which the rasterization obeys exactly). Each frame translates the
sprites by integer offsets so their appearance never changes.
"""
import os
import subprocess
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 40, 16
W = H = 1024
CENTER = (512, 512)

ORANGE = (245, 130, 48)
PINK = (255, 105, 180)


def extract_sprite(mask, cx, cy):
    """Return a full ring mask (odd square, centered) around (cx, cy).

    Uses the visible half-planes and mirror symmetry to fill in any part of
    the ring that was clipped by the image border.
    """
    ys, xs = np.nonzero(mask)
    r = max(cx - xs.min(), xs.max() - cx, cy - ys.min(), ys.max() - cy)
    # If clipped on some side, the opposite side gives the true radius.
    side = 2 * r + 1
    sp = np.zeros((side, side), bool)
    y0, x0 = cy - r, cx - r
    for y in range(side):
        for x in range(side):
            iy, ix = y0 + y, x0 + x
            if 0 <= iy < H and 0 <= ix < W:
                sp[y, x] = mask[iy, ix]
            else:  # mirror across the center (ring is symmetric)
                my, mx = 2 * cy - iy, 2 * cx - ix
                sp[y, x] = mask[my, mx]
    return sp, r


def ring_center(mask):
    ys, xs = np.nonzero(mask)
    cx = (xs.min() + xs.max()) // 2
    # vertical extent may be clipped: use the unclipped side
    top_clipped = ys.min() == 0
    bot_clipped = ys.max() == H - 1
    if top_clipped:
        cy = ys.max() - (xs.max() - xs.min()) // 2
    elif bot_clipped:
        cy = ys.min() + (xs.max() - xs.min()) // 2
    else:
        cy = (ys.min() + ys.max()) // 2
    return cx, cy


def paste(canvas, sprite, color, cx, cy):
    r = sprite.shape[0] // 2
    y0, x0 = cy - r, cx - r
    ys0, xs0 = max(0, -y0), max(0, -x0)
    ys1 = sprite.shape[0] - max(0, y0 + sprite.shape[0] - H)
    xs1 = sprite.shape[1] - max(0, x0 + sprite.shape[1] - W)
    sub = sprite[ys0:ys1, xs0:xs1]
    region = canvas[y0 + ys0:y0 + ys1, x0 + xs0:x0 + xs1]
    region[sub] = color


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)


def main():
    first = np.array(Image.open(FIRST).convert("RGB"))
    bg = first.copy()
    circles = []
    for color in (ORANGE, PINK):
        mask = np.all(first == color, axis=2)
        bg[mask] = 255  # background is uniform white behind the circles
        cx, cy = ring_center(mask)
        sprite, _ = extract_sprite(mask, cx, cy)
        circles.append((color, sprite, (cx, cy)))

    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for i in range(N_FRAMES):
        t = ease(i / (N_FRAMES - 1))
        frame = bg.copy()
        # Draw larger circle first so the smaller one stays on top.
        for color, sprite, (cx, cy) in sorted(circles, key=lambda c: -c[1].shape[0]):
            x = int(round(cx + (CENTER[0] - cx) * t))
            y = int(round(cy + (CENTER[1] - cy) * t))
            paste(frame, sprite, color, x, y)
        frames.append(frame)

    assert np.array_equal(frames[0], first), "first frame must match input"
    raw = np.stack(frames).tobytes()
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "slow", "-crf", "12",
        "-movflags", "+faststart", OUT,
    ]
    subprocess.run(cmd, input=raw, check=True)
    Image.fromarray(frames[-1]).save(os.path.join(OUT_DIR, "last_frame.png"))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
