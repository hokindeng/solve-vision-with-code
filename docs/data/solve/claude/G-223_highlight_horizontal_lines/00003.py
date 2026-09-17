#!/usr/bin/env python3
"""Circle all horizontal lines in first_frame.png with black circles, animated."""
import os, subprocess, math
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 48
SS = 4  # supersampling for anti-aliased circle strokes


def find_lines(img):
    """Group non-background pixels by exact color; return list of (color, bbox)."""
    arr = np.array(img.convert("RGB"))
    bg = arr[0, 0]
    mask = np.any(arr != bg, axis=2)
    cols, inv = np.unique(arr[mask].reshape(-1, 3), axis=0, return_inverse=True)
    ys, xs = np.nonzero(mask)
    lines = []
    for i, c in enumerate(cols):
        sel = inv.ravel() == i
        if sel.sum() < 50:
            continue
        x0, x1 = xs[sel].min(), xs[sel].max()
        y0, y1 = ys[sel].min(), ys[sel].max()
        lines.append((tuple(int(v) for v in c), (x0, y0, x1, y1)))
    return lines


def horizontal(bbox):
    x0, y0, x1, y1 = bbox
    return (x1 - x0) > 3 * (y1 - y0)


def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * min(max(t, 0.0), 1.0))


def draw_arc_frame(base, ellipses, progress):
    """Draw each ellipse as a partial arc (0..progress fraction), supersampled."""
    W, H = base.size
    layer = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for (cx, cy, rx, ry), p in zip(ellipses, progress):
        if p <= 0:
            continue
        box = [(cx - rx) * SS, (cy - ry) * SS, (cx + rx) * SS, (cy + ry) * SS]
        start = -90  # begin at top, sweep clockwise
        end = start + 360 * p
        if p >= 1:
            d.ellipse(box, outline=(0, 0, 0, 255), width=4 * SS)
        else:
            d.arc(box, start, end, fill=(0, 0, 0, 255), width=4 * SS)
    layer = layer.resize((W, H), Image.LANCZOS)
    frame = base.convert("RGBA")
    frame.alpha_composite(layer)
    return frame.convert("RGB")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = Image.open(SRC).convert("RGB")
    lines = find_lines(base)
    targets = [b for _, b in lines if horizontal(b)]
    ellipses = []
    for x0, y0, x1, y1 in targets:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rx = (x1 - x0) / 2 + 28
        ry = max((y1 - y0) / 2 + 28, 0.28 * rx)
        ellipses.append((cx, cy, rx, ry))
    print(f"lines: {len(lines)}, horizontal: {len(targets)} -> {ellipses}")

    n = max(len(ellipses), 1)
    # Stagger circles across frames 1..N-2 so first frame is untouched and
    # the final frames show the completed result.
    t_start, t_end = 1 / (N_FRAMES - 1), (N_FRAMES - 4) / (N_FRAMES - 1)
    span = (t_end - t_start)
    per = span / (1 + 0.6 * (n - 1))  # overlap successive circles

    frames_dir = os.path.join(OUT_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for f in range(N_FRAMES):
        t = f / (N_FRAMES - 1)
        prog = []
        for i in range(n):
            s = t_start + i * per * 0.6
            prog.append(ease((t - s) / per))
        frame = base if f == 0 else draw_arc_frame(base, ellipses, prog)
        frame.save(os.path.join(frames_dir, f"{f:04d}.png"))

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(frames_dir, "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT,
    ], check=True)
    for fn in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, fn))
    os.rmdir(frames_dir)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
