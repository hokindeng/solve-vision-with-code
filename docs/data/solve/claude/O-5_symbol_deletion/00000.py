#!/usr/bin/env python3
"""Symbol deletion: erase the red-bordered target symbol from the sequence.

The remaining symbols keep their exact positions; only the pixels inside the
red-bordered cell change. The target shrinks and fades to the background over
the middle of the clip, with short holds at both ends.
"""
import os
import subprocess
import numpy as np
from PIL import Image

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS = 16
N_FRAMES = 46
HOLD_START = 6
HOLD_END = 6


def find_target_box(a):
    """Bounding box of the pure-red (255,0,0) rectangular border."""
    m = (a[:, :, 0] == 255) & (a[:, :, 1] == 0) & (a[:, :, 2] == 0)
    ys, xs = np.where(m)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1  # x0,y0,x1,y1 (exclusive)


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)


def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    H, W, _ = base.shape
    x0, y0, x1, y1 = find_target_box(base)
    bg = base[0, 0].copy()
    patch = Image.fromarray(base[y0:y1, x0:x1])
    pw, ph = patch.size

    os.makedirs(OUT_DIR, exist_ok=True)
    n_anim = N_FRAMES - HOLD_START - HOLD_END
    frames = []
    for i in range(N_FRAMES):
        if i < HOLD_START:
            t = 0.0
        elif i >= N_FRAMES - HOLD_END:
            t = 1.0
        else:
            t = ease((i - HOLD_START + 1) / n_anim)
        frame = base.copy()
        if t <= 0.0:
            frames.append(frame)
            continue
        region = np.empty((ph, pw, 3), dtype=np.float32)
        region[:] = bg
        s = 1.0 - t
        alpha = 1.0 - t
        if s > 0.01 and alpha > 0.0:
            nw, nh = max(1, int(round(pw * s))), max(1, int(round(ph * s)))
            small = np.array(patch.resize((nw, nh), Image.LANCZOS)).astype(np.float32)
            ox, oy = (pw - nw) // 2, (ph - nh) // 2
            sub = region[oy:oy + nh, ox:ox + nw]
            region[oy:oy + nh, ox:ox + nw] = sub * (1 - alpha) + small * alpha
        frame[y0:y1, x0:x1] = np.clip(region + 0.5, 0, 255).astype(np.uint8)
        frames.append(frame)

    # Encode with ffmpeg via raw RGB pipe.
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "medium",
        "-r", str(FPS), OUT,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {OUT}: {len(frames)} frames @ {FPS} fps, target box {(x0, y0, x1, y1)}")


if __name__ == "__main__":
    main()
