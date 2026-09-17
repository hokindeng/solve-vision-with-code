"""Generate /app/output/video.mp4: the bottom-row rectangle first scales up
(same factor as the example cross, 165/135), then its outline thickens from
2 px to 6 px (as in the example's third cross). Everything else is untouched."""
import os
import subprocess
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

N_FRAMES, FPS = 16, 16
GREEN = np.array([0, 100, 0], np.uint8)

# Original rectangle: pixels x 106..240, y 666..698, stroke 2 (plus 4 stray
# join pixels on rows 665/699). Stroke centre-line in continuous coords:
CX, CY = 173.5, 682.5          # centre of the shape
HW0, HH0 = 66.5, 15.5          # half extents of the stroke centre-line
W0, W1 = 2.0, 6.0              # stroke width: thin -> thick
SCALE = 165.0 / 135.0          # measured from the example cross (135 -> 165)
ERASE = (80, 652, 267, 713)  # clears original + largest rendered rectangle (x0,y0,x1,y1)


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)


def rect_outline_mask(shape, cx, cy, hw, hh, width):
    """Boolean mask of a rectangle outline whose stroke (given width) is
    centred on the rectangle path with half extents (hw, hh)."""
    h, w = shape
    half = width / 2.0
    ox0, ox1 = int(round(cx - hw - half)), int(round(cx + hw + half))
    oy0, oy1 = int(round(cy - hh - half)), int(round(cy + hh + half))
    ix0, ix1 = int(round(cx - hw + half)), int(round(cx + hw - half))
    iy0, iy1 = int(round(cy - hh + half)), int(round(cy + hh - half))
    m = np.zeros(shape, bool)
    m[oy0:oy1, ox0:ox1] = True
    m[iy0:iy1, ix0:ix1] = False
    return m


def render(base, k):
    """Frame k of N_FRAMES. Phase 1 (frames 0..8): scale. Phase 2 (8..15): stroke."""
    if k == 0:
        return base.copy()
    split = 8
    if k <= split:
        s = 1.0 + (SCALE - 1.0) * ease(k / split)
        width = W0
    else:
        s = SCALE
        width = W0 + (W1 - W0) * ease((k - split) / (N_FRAMES - 1 - split))
    width = float(round(width))  # integer stroke widths keep edges crisp
    frame = base.copy()
    x0, y0, x1, y1 = ERASE
    frame[y0:y1, x0:x1] = 255
    m = rect_outline_mask(frame.shape[:2], CX, CY, HW0 * s, HH0 * s, width)
    frame[m] = GREEN
    return frame


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = np.array(Image.open(FIRST).convert("RGB"))
    frames = [render(base, k) for k in range(N_FRAMES)]
    for k, f in enumerate(frames):
        Image.fromarray(f).save(os.path.join(OUT_DIR, f"frame_{k:02d}.png"))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "1024x1024", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
           "-r", str(FPS), OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode:
        raise SystemExit("ffmpeg failed")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
