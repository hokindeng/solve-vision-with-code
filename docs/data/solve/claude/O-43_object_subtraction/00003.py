#!/usr/bin/env python3
"""Generate a video that removes the blue cone from first_frame.png.

The blue cone (a downward-pointing blue triangle) is faded smoothly into the
white background over the full 6 s duration. Every other pixel is copied from
the first frame unchanged in every frame.
"""
import os
import subprocess
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS = 16
N_FRAMES = 96


def blue_mask(img):
    """Pixels belonging to the blue cone: strongly blue-dominant."""
    r, g, b = (img[..., i].astype(int) for i in range(3))
    return (b > r + 40) & (b > g + 40)


def background_color(img, mask):
    """Colour surrounding the cone (sampled just outside its bounding box)."""
    ys, xs = np.where(mask)
    y0, y1 = max(ys.min() - 3, 0), min(ys.max() + 4, img.shape[0])
    x0, x1 = max(xs.min() - 3, 0), min(xs.max() + 4, img.shape[1])
    region = img[y0:y1, x0:x1]
    ring = ~mask[y0:y1, x0:x1]
    vals, counts = np.unique(region[ring].reshape(-1, 3), axis=0, return_counts=True)
    return vals[np.argmax(counts)].astype(np.float32)


def ease(t):
    """Smooth ease-in-out (smoothstep)."""
    return t * t * (3.0 - 2.0 * t)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = np.array(Image.open(SRC).convert("RGB"))
    mask = blue_mask(base)
    bg = background_color(base, mask)

    cone_pixels = base[mask].astype(np.float32)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{base.shape[1]}x{base.shape[0]}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-preset", "slow", "-crf", "8",
        "-pix_fmt", "yuv420p", "-r", str(FPS), "-movflags", "+faststart",
        OUT,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)          # 0 on first frame, 1 on last
        a = ease(t)                      # cone opacity goes 1 -> 0
        frame = base.copy()
        blended = cone_pixels * (1.0 - a) + bg * a
        frame[mask] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
