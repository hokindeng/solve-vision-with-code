#!/usr/bin/env python3
"""Draw one red circle, step by step, around the largest number (87)."""
import math, os, subprocess
import numpy as np
from PIL import Image, ImageDraw

BASE = "/app/first_frame.png"
OUT_DIR = "/app/output"
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 80
SS = 4  # supersampling for anti-aliased stroke

# Numbers on the canvas (value, glyph bounding box measured from first_frame.png)
NUMBERS = {
    15: (228, 143, 372, 235),
    83: (570, 226, 734, 317),   # upper 83
    87: (459, 318, 608, 400),
    44: (690, 335, 845, 420),
    31: (455, 505, 610, 595),
    83.0001: (605, 680, 760, 770),  # lower 83 (distinct key only)
}
largest = max(NUMBERS, key=lambda k: float(k))
x0, y0, x1, y1 = NUMBERS[largest]
cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
RX, RY = (x1 - x0) / 2.0 + 38, (y1 - y0) / 2.0 + 28   # oval "circle" enclosing the whole glyph
STROKE = 6
RED = (220, 20, 20)

# timeline: hold (compare) -> draw arc -> hold complete
HOLD_START, DRAW_END = 14, 68


def stroke_mask(frac):
    """Alpha mask (0..255) of the arc covering `frac` of the full ellipse, starting at top, clockwise."""
    if frac <= 0:
        return None
    W = H = 1024 * SS
    layer = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(layer)
    bbox = [(cx - RX) * SS, (cy - RY) * SS, (cx + RX) * SS, (cy + RY) * SS]
    start = -90.0
    end = start + 360.0 * min(frac, 1.0)
    if frac >= 1.0:
        d.ellipse(bbox, outline=255, width=STROKE * SS)
    else:
        d.arc(bbox, start=start, end=end, fill=255, width=STROKE * SS)
        # round caps
        for ang in (start, end):
            t = math.radians(ang)
            px, py = (cx + RX * math.cos(t)) * SS, (cy + RY * math.sin(t)) * SS
            r = STROKE * SS / 2.0
            d.ellipse([px - r, py - r, px + r, py + r], fill=255)
    layer = layer.resize((1024, 1024), Image.LANCZOS)
    return np.asarray(layer, dtype=np.float32) / 255.0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = np.asarray(Image.open(BASE).convert("RGB"), dtype=np.float32)
    red = np.array(RED, dtype=np.float32)[None, None, :]

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", "1024x1024", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-r", str(FPS), OUT],
        stdin=subprocess.PIPE)
    for i in range(N_FRAMES):
        if i < HOLD_START:
            frac = 0.0
        elif i >= DRAW_END:
            frac = 1.0
        else:
            t = (i - HOLD_START) / float(DRAW_END - HOLD_START)
            frac = 0.5 - 0.5 * math.cos(math.pi * t)   # ease in/out
        frame = base.copy()
        m = stroke_mask(frac)
        if m is not None:
            a = m[..., None]
            frame = frame * (1 - a) + red * a
        ff.stdin.write(np.clip(frame + 0.5, 0, 255).astype(np.uint8).tobytes())
    ff.stdin.close()
    ff.wait()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
