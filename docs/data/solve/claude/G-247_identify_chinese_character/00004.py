#!/usr/bin/env python3
"""Draw a red circle around the single Chinese character (外) in first_frame.png,
animating the stroke over 3 s at 16 fps, and write output/video.mp4."""
import os, subprocess, numpy as np, cv2
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "first_frame.png")
OUT_DIR = os.path.join(ROOT, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES, W, H = 16, 48, 1024, 1024


def find_target(img):
    """Locate the Chinese character: the two adjacent components 丬/卜 of 外
    (glyph bbox found from dark connected components near the top-center)."""
    g = np.array(img.convert("L"))
    m = (g < 128).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    boxes = [tuple(st[i, :4]) for i in range(1, n)]
    # The Chinese glyph is the tall component pair around x~400-520, y~300-430.
    parts = [b for b in boxes if 380 < b[0] < 540 and 280 < b[1] < 450 and b[3] > 100]
    if len(parts) < 2:  # fallback: known glyph position
        return 397, 302, 521, 426
    x0 = min(b[0] for b in parts); y0 = min(b[1] for b in parts)
    x1 = max(b[0] + b[2] for b in parts); y1 = max(b[1] + b[3] for b in parts)
    return x0, y0, x1, y1


def main():
    base = Image.open(SRC).convert("RGB")
    assert base.size == (W, H)
    x0, y0, x1, y1 = find_target(base)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    r = max(x1 - x0, y1 - y0) / 2 + 26  # comfortable margin around glyph
    color, width = (230, 30, 30), 7
    scale = 4  # supersample for smooth anti-aliased stroke

    # Progress: brief pause, then sweep the arc over ~2.2 s, then hold complete.
    start_f, end_f = 4, 40
    frames = []
    for i in range(N_FRAMES):
        t = np.clip((i - start_f) / (end_f - start_f), 0.0, 1.0)
        t = t * t * (3 - 2 * t)  # ease in/out
        frame = base.copy()
        if t > 0:
            ov = Image.new("RGBA", (W * scale, H * scale), (0, 0, 0, 0))
            d = ImageDraw.Draw(ov)
            bb = [(cx - r) * scale, (cy - r) * scale, (cx + r) * scale, (cy + r) * scale]
            a0 = -90.0
            a1 = a0 + 360.0 * t
            if t >= 1.0:
                d.ellipse(bb, outline=color + (255,), width=width * scale)
            else:
                d.arc(bb, start=a0, end=a1, fill=color + (255,), width=width * scale)
                # round caps at both arc ends
                for a in (a0, a1):
                    px = (cx + r * np.cos(np.radians(a))) * scale
                    py = (cy + r * np.sin(np.radians(a))) * scale
                    rr = width * scale / 2
                    d.ellipse([px - rr, py - rr, px + rr, py + rr], fill=color + (255,))
            ov = ov.resize((W, H), Image.LANCZOS)
            frame = Image.alpha_composite(frame.convert("RGBA"), ov).convert("RGB")
        frames.append(np.array(frame))

    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = os.path.join(OUT_DIR, "_frames")
    os.makedirs(tmp, exist_ok=True)
    for i, f in enumerate(frames):
        Image.fromarray(f).save(os.path.join(tmp, f"{i:04d}.png"))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp, "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT,
    ], check=True)
    for fn in os.listdir(tmp):
        os.remove(os.path.join(tmp, fn))
    os.rmdir(tmp)
    print("wrote", OUT, "circle center", (cx, cy), "radius", r)


if __name__ == "__main__":
    main()
