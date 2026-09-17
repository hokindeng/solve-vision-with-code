#!/usr/bin/env python3
"""Mark the longest side of the polygon in first_frame.png with a small red
circle at its midpoint, animated over ~25 frames, and write output/video.mp4."""
import os, subprocess, math
import numpy as np
import cv2
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 25
RADIUS = 9          # "small" red circle
RED = (255, 0, 0)
SS = 4              # supersampling for smooth edges


def polygon_vertices(img_bgr):
    """Step 1: recover the polygon's 8 vertices from the image."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray < 250).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    eps = 1.0
    while True:
        ap = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
        if len(ap) <= 8:
            break
        eps += 0.5
    return [tuple(map(float, p)) for p in ap]


def longest_edge(verts):
    """Step 2: compare all edge lengths, return the longest one and its midpoint."""
    edges = []
    for i in range(len(verts)):
        a, b = verts[i], verts[(i + 1) % len(verts)]
        edges.append((math.dist(a, b), a, b))
    for L, a, b in edges:
        print(f"edge {a} -> {b}: length {L:.1f}")
    L, a, b = max(edges, key=lambda e: e[0])
    mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    print(f"longest: {a} -> {b} ({L:.1f}), midpoint {mid}")
    return a, b, mid


def draw_circle(base, center, r):
    """Composite an anti-aliased filled red circle of radius r onto base (PIL RGB)."""
    if r <= 0:
        return base.copy()
    cx, cy = center
    x0, y0 = int(cx - r - 2), int(cy - r - 2)
    x1, y1 = int(cx + r + 3), int(cy + r + 3)
    w, h = x1 - x0, y1 - y0
    m = Image.new("L", (w * SS, h * SS), 0)
    d = ImageDraw.Draw(m)
    d.ellipse([(cx - x0 - r) * SS, (cy - y0 - r) * SS,
               (cx - x0 + r) * SS, (cy - y0 + r) * SS], fill=255)
    m = m.resize((w, h), Image.LANCZOS)
    out = base.copy()
    patch = Image.new("RGB", (w, h), RED)
    out.paste(patch, (x0, y0), m)
    return out


def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * t)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    bgr = cv2.imread(SRC)
    verts = polygon_vertices(bgr)
    _, _, mid = longest_edge(verts)
    base = Image.open(SRC).convert("RGB")

    # Step 3: animate the red circle appearing at the midpoint.
    hold_start, hold_end = 3, 4
    grow = N_FRAMES - hold_start - hold_end
    frames = []
    for i in range(N_FRAMES):
        if i < hold_start:
            r = 0.0
        elif i >= N_FRAMES - hold_end:
            r = RADIUS
        else:
            t = (i - hold_start + 1) / grow
            r = RADIUS * ease(t)
        frames.append(draw_circle(base, mid, r))

    tmp = os.path.join(OUT_DIR, "_frames")
    os.makedirs(tmp, exist_ok=True)
    for i, f in enumerate(frames):
        f.save(os.path.join(tmp, f"f{i:03d}.png"))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", os.path.join(tmp, "f%03d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
                    "-preset", "slow", OUT], check=True)
    for fn in os.listdir(tmp):
        os.remove(os.path.join(tmp, fn))
    os.rmdir(tmp)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
