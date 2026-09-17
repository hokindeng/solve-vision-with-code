#!/usr/bin/env python3
"""Mark the longest side of the polygon in first_frame.png with a small red
circle at its midpoint, animated over ~25 frames at 16 fps."""
import os, subprocess, math
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 25, 16
N_SIDES = 5
RED = (0, 0, 255)  # BGR
RADIUS, THICK = 12, 3


def find_polygon(img):
    mask = (np.abs(img.astype(int) - 255).sum(2) > 10).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    for eps in np.arange(0.5, 20, 0.5):
        ap = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(ap) == N_SIDES:
            return ap.astype(float)
    return cv2.approxPolyDP(c, 3, True).reshape(-1, 2).astype(float)


def longest_edge(pts):
    best, bl = None, -1
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        L = math.hypot(*(b - a))
        if L > bl:
            bl, best = L, (a, b)
    return best, bl


def draw_arc(img, center, radius, frac):
    """Draw a red circular arc that sweeps `frac` (0..1) of the full circle."""
    if frac <= 0:
        return
    if frac >= 1:
        cv2.circle(img, center, radius, RED, THICK, cv2.LINE_AA)
        return
    cv2.ellipse(img, center, (radius, radius), -90, 0, 360 * frac, RED, THICK, cv2.LINE_AA)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = cv2.imread(SRC)
    pts = find_polygon(base)
    (a, b), L = longest_edge(pts)
    mid = ((a + b) / 2).round().astype(int)
    print("vertices:", pts.astype(int).tolist())
    print("longest edge:", a.astype(int).tolist(), b.astype(int).tolist(), "len=%.1f" % L, "mid=", mid.tolist())

    # Timeline: frame 0 untouched; the circle is stroked progressively over
    # frames 1..20 then held complete for the remaining frames.
    sweep_start, sweep_end = 1, 20
    frames = []
    for f in range(N_FRAMES):
        img = base.copy()
        if f >= sweep_start:
            t = min(1.0, (f - sweep_start + 1) / (sweep_end - sweep_start + 1))
            t = t * t * (3 - 2 * t)  # ease in/out
            draw_arc(img, tuple(int(v) for v in mid), RADIUS, t)
        frames.append(img)

    h, w = base.shape[:2]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(fr.tobytes())
    p.stdin.close(); p.wait()
    cv2.imwrite(os.path.join(OUT_DIR, "last_frame.png"), frames[-1])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
