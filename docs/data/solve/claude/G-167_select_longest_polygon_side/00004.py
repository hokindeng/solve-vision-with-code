#!/usr/bin/env python3
"""Mark the longest side of the 5-gon in first_frame.png with a small red circle
at its midpoint, animated step by step, and encode as H.264 video."""
import os, subprocess, math
import numpy as np
import cv2
from PIL import Image, ImageDraw

APP = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES, N_SIDES = 16, 25, 5
RED = (220, 30, 30)
RADIUS = 11          # small red circle
SS = 4               # supersampling for smooth edges

def find_polygon(img):
    arr = np.asarray(img).astype(int)
    mask = ((np.abs(arr - 255).sum(2)) > 30).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    for eps in np.arange(0.5, 20, 0.5):
        ap = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(ap) == N_SIDES:
            return [tuple(map(float, p)) for p in ap]
    raise RuntimeError("could not find a %d-gon" % N_SIDES)

def longest_edge(pts):
    edges = [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
    lengths = [math.dist(a, b) for a, b in edges]
    for (a, b), L in zip(edges, lengths):
        print("edge %s -> %s : %.1f" % (a, b, L))
    i = int(np.argmax(lengths))
    return edges[i]

def draw_marker(base, cx, cy, t):
    """t in [0,1]: 0..0.6 sweep an outline arc, 0.6..1 fill the disc."""
    if t <= 0:
        return base.copy()
    W, H = base.size
    layer = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    r = RADIUS * SS
    box = [cx * SS - r, cy * SS - r, cx * SS + r, cy * SS + r]
    w = max(2, int(2.5 * SS))
    if t < 0.6:
        ang = 360 * (t / 0.6)
        d.arc(box, start=-90, end=-90 + ang, fill=RED + (255,), width=w)
    else:
        d.ellipse(box, outline=RED + (255,), width=w)
        f = (t - 0.6) / 0.4
        rf = (r - w) * f
        if rf > 0:
            d.ellipse([cx * SS - rf, cy * SS - rf, cx * SS + rf, cy * SS + rf],
                      fill=RED + (255,))
    layer = layer.resize((W, H), Image.LANCZOS)
    out = base.convert("RGBA")
    out.alpha_composite(layer)
    return out.convert("RGB")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = Image.open(SRC).convert("RGB")
    pts = find_polygon(base)
    print("vertices:", pts)
    a, b = longest_edge(pts)
    cx, cy = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    print("longest edge %s -> %s, midpoint (%.1f, %.1f)" % (a, b, cx, cy))

    frames = []
    hold_start, hold_end = 3, 3   # frames of "reading/comparing" and of final result
    active = N_FRAMES - hold_start - hold_end
    for i in range(N_FRAMES):
        if i < hold_start:
            t = 0.0
        elif i >= N_FRAMES - hold_end:
            t = 1.0
        else:
            t = (i - hold_start + 1) / active
        frames.append(draw_marker(base, cx, cy, t))
    frames[0] = base.copy()

    tmp = os.path.join(OUT_DIR, "_frames")
    os.makedirs(tmp, exist_ok=True)
    for i, f in enumerate(frames):
        f.save(os.path.join(tmp, "f%03d.png" % i))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", os.path.join(tmp, "f%03d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
                    "-r", str(FPS), OUT], check=True)
    for fn in os.listdir(tmp):
        os.remove(os.path.join(tmp, fn))
    os.rmdir(tmp)
    print("wrote", OUT)

if __name__ == "__main__":
    main()
