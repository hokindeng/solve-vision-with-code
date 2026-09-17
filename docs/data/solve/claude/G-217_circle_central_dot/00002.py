#!/usr/bin/env python3
"""Circle the middle-by-count dot in first_frame.png with a red circle, animated over 22 frames."""
import os
import subprocess
import numpy as np
import cv2
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS = 16
N_FRAMES = 22
RED = (255, 0, 0)


def find_dots(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = (gray < 128).astype(np.uint8)
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask)
    dots = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 20:
            continue
        dots.append((cent[i][0], cent[i][1], max(w, h) / 2.0))
    dots.sort(key=lambda d: d[0])
    return dots


def main():
    base = np.array(Image.open(SRC).convert("RGB"))
    dots = find_dots(base)
    if len(dots) % 2 == 0:
        raise SystemExit(f"Even number of dots ({len(dots)}); no single middle dot.")
    cx, cy, r = dots[len(dots) // 2]

    # Circle radius: comfortably outside the dot, but not touching neighbours.
    spacing = min(abs(dots[i + 1][0] - dots[i][0]) for i in range(len(dots) - 1))
    thickness = 3
    radius = int(min(r + 8, (spacing - r) - thickness - 2))

    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for f in range(N_FRAMES):
        frame = base.copy()
        if f > 0:
            t = f / (N_FRAMES - 1)  # 0..1, frame 0 is untouched
            sweep = 360.0 * t
            # Supersample a patch around the circle for a smooth arc.
            S = 4
            pad = radius + thickness + 4
            x0, y0 = int(cx) - pad, int(cy) - pad
            patch = frame[y0:y0 + 2 * pad, x0:x0 + 2 * pad]
            big = cv2.resize(patch, None, fx=S, fy=S, interpolation=cv2.INTER_NEAREST)
            center = (int(round((cx - x0) * S)), int(round((cy - y0) * S)))
            if sweep >= 359.5:
                cv2.circle(big, center, radius * S, RED, thickness * S, lineType=cv2.LINE_AA)
            else:
                cv2.ellipse(big, center, (radius * S, radius * S), 0, -90, -90 + sweep,
                            RED, thickness * S, lineType=cv2.LINE_AA)
            small = cv2.resize(big, (2 * pad, 2 * pad), interpolation=cv2.INTER_AREA)
            frame[y0:y0 + 2 * pad, x0:x0 + 2 * pad] = small
        frames.append(frame)

    h, w = base.shape[:2]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(np.ascontiguousarray(fr).tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {OUT}: {len(frames)} frames, circle at ({cx:.0f},{cy:.0f}) r={radius}")


if __name__ == "__main__":
    main()
