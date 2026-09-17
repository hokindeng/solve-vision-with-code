#!/usr/bin/env python3
"""Remove the green cone and the orange cone from first_frame.png over a 6 s video.

The two cones (apex-down triangles: green at bottom-right, orange at bottom-left)
are dissolved into the white background one after the other, with a short overlap.
Every pixel outside the cones' footprints is copied unchanged from the first frame.
"""
import subprocess, os
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
W = H = 1024
FPS = 16
N = 96

src = np.array(Image.open(SRC).convert("RGB")).astype(np.float32)
bg = np.array([255.0, 255.0, 255.0], dtype=np.float32)  # background is pure white

def soft_mask(box, color):
    """Per-pixel alpha (0..1) of how much of `color` is present inside box,
    derived from the distance to white so anti-aliased edges fade correctly."""
    x0, y0, x1, y1 = box
    m = np.zeros((H, W), dtype=np.float32)
    patch = src[y0:y1, x0:x1]
    # fraction along the white -> color line, measured on the channel with the biggest change
    diff = bg - np.array(color, dtype=np.float32)
    ch = int(np.argmax(np.abs(diff)))
    frac = (bg[ch] - patch[:, :, ch]) / diff[ch]
    m[y0:y1, x0:x1] = np.clip(frac, 0.0, 1.0)
    return m

# Footprints (bounding boxes padded by a few px) and fill colours measured from the frame.
green_cone = soft_mask((610, 766, 800, 955), (0, 200, 0))     # bottom-right, apex down
orange_cone = soft_mask((70, 741, 258, 930), (255, 128, 0))   # bottom-left, apex down

def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)

def progress(i, start, end):
    return smoothstep((i - start) / float(end - start))

def render(i):
    # Green cone dissolves during frames 4..52, orange cone during 44..92.
    g = progress(i, 4, 52)
    o = progress(i, 44, 92)
    removal = np.clip(green_cone * g + orange_cone * o, 0.0, 1.0)[:, :, None]
    frame = src * (1.0 - removal) + bg * removal
    return np.clip(frame + 0.5, 0, 255).astype(np.uint8)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "medium",
           "-movflags", "+faststart", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(N):
        p.stdin.write(render(i).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise SystemExit("ffmpeg failed")
    print("wrote", OUT)

if __name__ == "__main__":
    main()
