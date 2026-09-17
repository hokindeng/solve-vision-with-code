#!/usr/bin/env python3
"""Subtractive (multiply) pigment mixing: fill the central mixing zone with c1*c2/255."""
import subprocess
from collections import deque

import numpy as np
from PIL import Image

W = H = 1024
FPS = 16
N_FRAMES = 44
LEFT = np.array([25, 54, 147], dtype=np.int64)
RIGHT = np.array([163, 122, 58], dtype=np.int64)


def mixed_color(c1, c2):
    return np.round(c1 * c2 / 255.0).astype(np.uint8)


def zone_interior_mask(img):
    """Flood-fill from the canvas centre over non-black pixels; the black border stops it."""
    black = np.all(img < 40, axis=2)
    h, w = black.shape
    mask = np.zeros((h, w), dtype=bool)
    cy, cx = h // 2, w // 2
    assert not black[cy, cx]
    q = deque([(cy, cx)])
    mask[cy, cx] = True
    while q:
        y, x = q.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and not mask[ny, nx] and not black[ny, nx]:
                mask[ny, nx] = True
                q.append((ny, nx))
    return mask


def main():
    base = np.array(Image.open("/app/first_frame.png").convert("RGB"))
    mixed = mixed_color(LEFT, RIGHT)
    interior = zone_interior_mask(base)
    ys, xs = np.where(interior)
    cy, cx = ys.mean(), xs.mean()
    dist = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    max_d = dist.max()

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "medium",
        "/app/output/video.mp4",
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)  # 0 -> 1
        frame = base.copy()
        if i > 0:
            # ease-in-out radial wipe from the zone centre; fully covered on the last frame
            s = 0.5 - 0.5 * np.cos(np.pi * t)
            sel = dist <= s * (max_d + 1.0)
            frame[ys[sel], xs[sel]] = mixed
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    print("mixed colour:", tuple(int(v) for v in mixed), "interior pixels:", int(interior.sum()))


if __name__ == "__main__":
    main()
