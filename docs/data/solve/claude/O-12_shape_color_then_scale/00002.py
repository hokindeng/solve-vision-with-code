#!/usr/bin/env python3
"""Generate the analogy-completion video: octagon gets recolored, then resized."""
import math, os, subprocess, shutil, tempfile
import numpy as np
from PIL import Image, ImageDraw

BASE = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
N_FRAMES, FPS, SS = 60, 16, 4

COLOR_333 = (80, 80, 229)     # starting blue
COLOR_295 = (53, 103, 153)    # target teal
OUTLINE = (0, 0, 0)
R_XL = 90.0                   # measured circumradius of the extra_large octagon
R_L = R_XL * 0.775            # large/extra_large ratio measured from the hearts
CY = 682                      # bottom-row centre y
CX_MID, CX_RIGHT = 518, 854   # bottom-row slot centres (match "?" centres)
# bounding boxes of the two "?" glyphs (x0, y0, x1, y1), slightly padded
Q_MID = (500, 655, 537, 709)
Q_RIGHT = (836, 655, 873, 709)


def smooth(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def seg(f, a, b):
    """Progress 0..1 of frame f over [a, b]."""
    return smooth((f - a) / float(b - a))


def lerp_color(c0, c1, t):
    return tuple(int(round(c0[i] + (c1[i] - c0[i]) * t)) for i in range(3))


def octagon_layer(cx, cy, r, fill, alpha):
    """Return an RGBA PIL image (full canvas) with a supersampled octagon."""
    pad = int(r) + 4
    x0, y0 = int(cx) - pad, int(cy) - pad
    size = 2 * pad
    big = Image.new("RGBA", (size * SS, size * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    pts = [((cx - x0 + r * math.sin(math.radians(45 * i))) * SS,
            (cy - y0 - r * math.cos(math.radians(45 * i))) * SS) for i in range(8)]
    d.polygon(pts, fill=fill + (255,), outline=OUTLINE + (255,), width=SS)
    small = big.resize((size, size), Image.LANCZOS)
    if alpha < 1.0:
        a = small.getchannel("A").point(lambda v: int(v * alpha))
        small.putalpha(a)
    return small, (x0, y0)


def fade_region(img, box, t):
    """Blend the region `box` of img toward white by fraction t."""
    if t <= 0:
        return
    x0, y0, x1, y1 = box
    reg = np.asarray(img.crop(box)).astype(np.float32)
    reg = reg + (255.0 - reg) * t
    img.paste(Image.fromarray(reg.round().astype(np.uint8)), (x0, y0))


def render(f, base):
    img = base.copy()
    # ---- Phase 1: middle slot -> recolor rule (frames 0..28)
    fade_region(img, Q_MID, seg(f, 2, 10))
    a1 = seg(f, 6, 14)
    if a1 > 0:
        col = lerp_color(COLOR_333, COLOR_295, seg(f, 14, 28))
        layer, pos = octagon_layer(CX_MID, CY, R_XL, col, a1)
        img.alpha_composite(layer, pos)
    # ---- Phase 2: right slot -> resize rule (frames 30..58)
    fade_region(img, Q_RIGHT, seg(f, 30, 38))
    a2 = seg(f, 34, 42)
    if a2 > 0:
        r = R_XL + (R_L - R_XL) * seg(f, 42, 57)
        layer, pos = octagon_layer(CX_RIGHT, CY, r, COLOR_295, a2)
        img.alpha_composite(layer, pos)
    return img.convert("RGB")


def main():
    base = Image.open(BASE).convert("RGBA")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = tempfile.mkdtemp()
    for f in range(N_FRAMES):
        render(f, base).save(os.path.join(tmp, "f%04d.png" % f))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp, "f%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT], check=True)
    shutil.rmtree(tmp)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
