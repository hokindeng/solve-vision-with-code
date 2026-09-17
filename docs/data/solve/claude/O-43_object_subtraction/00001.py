#!/usr/bin/env python3
"""Remove all orange objects from first_frame.png, fading them out over 6 s."""
import os
import subprocess
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 96


def orange_mask(img):
    """Pixels that are orange: high red, mid green, low blue."""
    r, g, b = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
    return (r > 200) & (g > 60) & (g < 200) & (b < 80)


def ease(t):
    # smoothstep for a gentle start and finish
    return t * t * (3 - 2 * t)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    src = np.array(Image.open(SRC).convert("RGB"))
    mask = orange_mask(src)
    # Background colour: the most common colour in the image (white here).
    cols, counts = np.unique(src.reshape(-1, 3), axis=0, return_counts=True)
    bg = cols[np.argmax(counts)].astype(np.float32)

    target = src.astype(np.float32).copy()
    target[mask] = bg  # final frame: orange objects replaced by background

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1024x1024", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-crf", "12", "-preset", "slow", "-pix_fmt", "yuv420p", "-r", str(FPS), OUT,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    srcf = src.astype(np.float32)
    for i in range(N_FRAMES):
        t = ease(i / (N_FRAMES - 1))
        frame = srcf.copy()
        # Only the orange pixels change; everything else stays as in the source.
        frame[mask] = (1 - t) * srcf[mask] + t * target[mask]
        proc.stdin.write(np.clip(frame + 0.5, 0, 255).astype(np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
