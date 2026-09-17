#!/usr/bin/env python3
"""Move the blue triangular agent along the shortest directed path
(green start -> top white node -> red end) and render an mp4."""
import os
import subprocess
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 30

GREEN = np.array([0, 128, 0], np.uint8)

# Node centres measured from first_frame.png (x, y).
START = (644, 800)   # green node
MID = (626, 286)     # top white node (edge start -> top, top -> red)
END = (322, 298)     # red node
PATH = [START, MID, END]


def main():
    frame0 = np.array(Image.open(FIRST).convert("RGB"))
    # Agent sprite = pure blue fill + dark-blue outline pixels.
    blue = (frame0[:, :, 0] == 0) & (frame0[:, :, 1] == 0) & (frame0[:, :, 2] >= 139)
    ys, xs = np.nonzero(blue)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sprite = frame0[y0:y1, x0:x1].copy()
    mask = blue[y0:y1, x0:x1]
    # Sprite anchor = its bbox centre; offset relative to start node centre.
    anchor = np.array([(x0 + x1 - 1) / 2.0, (y0 + y1 - 1) / 2.0])
    offset = anchor - np.array(START, float)

    # Background: original frame with the agent removed (it sits fully inside the green disc).
    bg = frame0.copy()
    bg[blue] = GREEN

    def ease(t):
        return 0.5 - 0.5 * np.cos(np.pi * t)

    # Timeline: hold 1 frame, hop 1 (13 frames), pause 2, hop 2 (12 frames), hold at end.
    segs = [(1, 14), (16, 28)]
    frames = []
    for f in range(N_FRAMES):
        if f == 0:
            frames.append(frame0.copy())
            continue
        pos = np.array(PATH[-1], float)
        for i, (a, b) in enumerate(segs):
            p, q = np.array(PATH[i], float), np.array(PATH[i + 1], float)
            if f < a:
                pos = p
                break
            if f <= b:
                t = ease((f - a) / (b - a))
                pos = p + (q - p) * t
                break
        img = bg.copy()
        cx, cy = np.round(pos + offset).astype(int)
        h, w = mask.shape
        tx, ty = cx - (w - 1) // 2, cy - (h - 1) // 2
        region = img[ty:ty + h, tx:tx + w]
        region[mask] = sprite[mask]
        frames.append(img)

    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = os.path.join(OUT_DIR, "_frames")
    os.makedirs(tmp, exist_ok=True)
    for i, fr in enumerate(frames):
        Image.fromarray(fr).save(os.path.join(tmp, f"{i:04d}.png"))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp, "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT,
    ], check=True)
    for fn in os.listdir(tmp):
        os.remove(os.path.join(tmp, fn))
    os.rmdir(tmp)
    print("wrote", OUT, len(frames), "frames")


if __name__ == "__main__":
    main()
