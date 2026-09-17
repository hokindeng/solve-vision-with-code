#!/usr/bin/env python3
"""Find the second-largest circle in first_frame.png and progressively circle it in red."""
import subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS = 16
N_FRAMES = 40
RED = (255, 0, 0)
GAP = 22          # distance between the shape's edge and the red ring
WIDTH = 8         # ring stroke width


def detect_circles(img):
    """Return list of (cx, cy, radius) for each non-white blob."""
    mask = (np.abs(img.astype(int) - 255).sum(2) > 30).astype(np.uint8)
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask)
    circles = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 50:
            continue
        circles.append((cent[i][0], cent[i][1], (w + h) / 4.0))
    return circles


def ease(t):
    return t * t * (3 - 2 * t)  # smoothstep


def main():
    base = Image.open(SRC).convert("RGB")
    circles = detect_circles(np.array(base))
    circles.sort(key=lambda c: -c[2])          # largest first
    cx, cy, r = circles[1]                     # second largest
    R = r + GAP
    bbox = [cx - R, cy - R, cx + R, cy + R]

    # Supersample the ring for a clean anti-aliased stroke.
    SS = 4
    frames = []
    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)
        sweep = 360.0 * ease(t)
        frame = base.copy()
        if sweep > 0:
            big = Image.new("RGBA", (base.width * SS, base.height * SS), (0, 0, 0, 0))
            d = ImageDraw.Draw(big)
            bb = [v * SS for v in bbox]
            start = -90
            if sweep >= 360:
                d.ellipse(bb, outline=RED + (255,), width=WIDTH * SS)
                caps = ()
            else:
                d.arc(bb, start=start, end=start + sweep, fill=RED + (255,), width=WIDTH * SS)
                caps = (start, start + sweep)
            # round caps on the arc ends
            for ang in caps:
                a = np.deg2rad(ang)
                px, py = (cx + R * np.cos(a)) * SS, (cy + R * np.sin(a)) * SS
                hw = WIDTH * SS / 2
                d.ellipse([px - hw, py - hw, px + hw, py + hw], fill=RED + (255,))
            overlay = big.resize(base.size, Image.LANCZOS)
            frame.paste(overlay, (0, 0), overlay)
        frames.append(np.array(frame))

    # Encode with ffmpeg: H.264, yuv420p.
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{base.width}x{base.height}",
           "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
           "-movflags", "+faststart", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"second largest circle at ({cx:.0f},{cy:.0f}) r={r:.0f}; wrote {OUT}")


if __name__ == "__main__":
    main()
