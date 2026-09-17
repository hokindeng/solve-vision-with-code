#!/usr/bin/env python3
"""Generate the analogy-completion video: shape colour changes, then scales."""
import math, subprocess, os
import numpy as np
from PIL import Image, ImageDraw

BASE = "/app/first_frame.png"
OUT_DIR = "/app/output"
OUT = os.path.join(OUT_DIR, "video.mp4")
W = H = 1024
FPS = 16
N = 60
SS = 4  # supersampling factor for antialiased polygon rendering

PURPLE = (132, 30, 153)   # color_394
OLIVE = (132, 153, 30)    # color_130
R_LARGE = 70.0
R_MEDIUM = 50.0           # medium/large = 100/140 (measured from top row)
ROW_Y = 682
SLOT2_X, SLOT3_X = 518, 854
Q1 = (505, 660, 532, 704)  # question-mark bounding boxes (x0,y0,x1,y1 exclusive)
Q2 = (841, 660, 868, 704)


def smooth(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def lerp_color(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def octagon_layer(cx, cy, r, fill):
    """Return (rgb uint8 HxWx3, alpha float HxW) of an outlined octagon."""
    big = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    pts = [((cx + r * math.cos(math.radians(a))) * SS + SS / 2 - 0.5,
            (cy + r * math.sin(math.radians(a))) * SS + SS / 2 - 0.5)
           for a in range(0, 360, 45)]
    d.polygon(pts, fill=fill + (255,), outline=(0, 0, 0, 255), width=SS)
    small = big.resize((W, H), Image.LANCZOS)
    arr = np.asarray(small).astype(np.float32)
    return arr[..., :3], arr[..., 3] / 255.0


def composite(canvas, rgb, alpha):
    a = alpha[..., None]
    return canvas * (1 - a) + rgb * a


def erase_box(canvas, base, box, t):
    """Fade the region `box` toward white by factor t."""
    x0, y0, x1, y1 = box
    region = base[y0:y1, x0:x1]
    canvas[y0:y1, x0:x1] = region * (1 - t) + 255.0 * t


def render_frame(i, base):
    canvas = base.astype(np.float32).copy()

    # ---- Phase 1 (frames 6-30): slot 2 gets the octagon, colour purple -> olive
    q1_fade = smooth((i - 6) / 8.0)
    oct2_in = smooth((i - 10) / 8.0)
    col_t = smooth((i - 18) / 12.0)
    # ---- Phase 2 (frames 30-56): slot 3 gets olive octagon, size large -> medium
    q2_fade = smooth((i - 30) / 8.0)
    oct3_in = smooth((i - 34) / 8.0)
    size_t = smooth((i - 42) / 14.0)

    if q1_fade > 0:
        erase_box(canvas, base, Q1, q1_fade)
    if q2_fade > 0:
        erase_box(canvas, base, Q2, q2_fade)

    if oct2_in > 0:
        rgb, a = octagon_layer(SLOT2_X, ROW_Y, R_LARGE, lerp_color(PURPLE, OLIVE, col_t))
        canvas = composite(canvas, rgb, a * oct2_in)
    if oct3_in > 0:
        r = R_LARGE + (R_MEDIUM - R_LARGE) * size_t
        rgb, a = octagon_layer(SLOT3_X, ROW_Y, r, OLIVE)
        canvas = composite(canvas, rgb, a * oct3_in)

    return np.clip(canvas + 0.5, 0, 255).astype(np.uint8)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = np.asarray(Image.open(BASE).convert("RGB"))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "medium", OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(N):
        frame = base if i == 0 else render_frame(i, base)
        proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    proc.stdin.close()
    proc.wait()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
