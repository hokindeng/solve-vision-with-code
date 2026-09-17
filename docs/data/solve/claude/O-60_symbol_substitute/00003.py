#!/usr/bin/env python3
"""Substitute the yellow right triangle at position 4 with an indigo solid heart.

Phase 1 (first half): triangle fades linearly into the white background.
Phase 2 (second half): heart fades in from white to indigo at the same cell.
Every other pixel is copied unchanged from first_frame.png.
"""
import subprocess
import numpy as np
from PIL import Image

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS = 16
N_FRAMES = 52
INDIGO = np.array([75, 0, 130], dtype=np.float32)
WHITE = np.array([255, 255, 255], dtype=np.float32)

base = np.array(Image.open(SRC).convert("RGB")).astype(np.float32)

# --- Cell 4 (border lines at x=517/613, y=464/560), interior only ---
x0, x1, y0, y1 = 518, 613, 465, 560
cell = base[y0:y1, x0:x1]
tri_mask = (cell[..., 0] > 200) & (cell[..., 1] > 200) & (cell[..., 2] < 100)
tri_color = np.array([255, 255, 0], dtype=np.float32)

# --- Heart shape: taken from the reference panel (top-right), same pixel size
# as the symbols in the row. Reference heart bbox: y 53..111, x 914..978.
ref = base[53:112, 914:979]
heart_shape = (ref[..., 2] > 100) & (ref[..., 0] < 100)
hh, hw = heart_shape.shape
# Place it centered horizontally in cell 4 (center x=565), same vertical
# placement as the existing heart in cell 5 (bbox y 488..546).
hx0 = 565 - hw // 2
hy0 = 488
heart_mask = np.zeros(base.shape[:2], dtype=bool)
heart_mask[hy0:hy0 + hh, hx0:hx0 + hw] = heart_shape

full_tri_mask = np.zeros(base.shape[:2], dtype=bool)
full_tri_mask[y0:y1, x0:x1] = tri_mask


def smooth(t):
    return t * t * (3 - 2 * t)


def make_frame(i):
    t = i / (N_FRAMES - 1)
    frame = base.copy()
    if t <= 0.5:
        a = smooth(t / 0.5)              # 0 -> 1 : triangle fades out
        frame[full_tri_mask] = tri_color * (1 - a) + WHITE * a
    else:
        a = smooth((t - 0.5) / 0.5)      # 0 -> 1 : heart fades in
        frame[full_tri_mask] = WHITE
        frame[heart_mask] = WHITE * (1 - a) + INDIGO * a
    return np.clip(np.round(frame), 0, 255).astype(np.uint8)


def main():
    frames = [make_frame(i) for i in range(N_FRAMES)]
    assert np.array_equal(frames[0], base.astype(np.uint8))
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1024x1024", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        raise SystemExit("ffmpeg failed")
    Image.fromarray(frames[-1]).save("/app/output/last_frame.png")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
